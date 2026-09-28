"""Audit whether learned latent state adds observable predictive information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from hh_self_discovery.blind_evaluation import load_latent_checkpoint
from hh_self_discovery.sampling import BalancedWindowSampler, ObservedDataset
from hh_self_discovery.training import ModeNormalizers, Standardizer, WindowTorchDataset


def normalizers(checkpoint: dict) -> ModeNormalizers:
    values = checkpoint["normalizers"]["current_clamp"]
    return ModeNormalizers(
        Standardizer(**values["command"]),
        Standardizer(**values["response"]),
    )


def summaries(history: np.ndarray, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h_i, h_v = history[..., 0], history[..., 1]
    instant = np.column_stack([
        h_i[:, -1], h_v[:, -1], command[:, 0],
        command.mean(1), command.std(1), command.min(1), command.max(1),
    ])
    quarter = max(2, history.shape[1] // 4)
    extended = np.column_stack([
        instant,
        h_i.mean(1), h_i.std(1), h_i.min(1), h_i.max(1),
        h_v.mean(1), h_v.std(1), h_v.min(1), h_v.max(1),
        h_v[:, -quarter:].mean(1) - h_v[:, :quarter].mean(1),
        h_i[:, -quarter:].mean(1) - h_i[:, :quarter].mean(1),
    ])
    return instant, extended


def collect(model, checkpoint, root: str, split: str, windows: int, seed: int, device):
    dataset = ObservedDataset(root, split)
    sampler = BalancedWindowSampler(
        dataset, checkpoint["history_steps"], checkpoint["prediction_steps"]
    )
    refs = sampler.sample_mode_epoch("current_clamp", windows, seed, strict=False)
    loader = DataLoader(
        WindowTorchDataset(sampler, refs, normalizers(checkpoint)),
        batch_size=128,
        shuffle=False,
    )
    instant_all, history_all, latent_all, labels, predicted_labels = [], [], [], [], []
    threshold = model.normalize(torch.tensor(0.0, device=device), "current_clamp", "response").item()
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            command = batch["future_command"].to(device)
            target = batch["target"].numpy()
            latent = model.encode(history, "current_clamp")
            prediction = model(history, command, "current_clamp")
            instant, extended = summaries(
                history.cpu().numpy(), command.cpu().numpy()
            )
            instant_all.append(instant)
            history_all.append(extended)
            latent_all.append(latent.cpu().numpy())
            labels.append((target.max(axis=1) >= threshold).astype(np.int64))
            predicted_labels.append(
                (prediction.cpu().numpy().max(axis=1) >= threshold).astype(np.int64)
            )
    return {
        "instant": np.concatenate(instant_all),
        "history": np.concatenate(history_all),
        "latent": np.concatenate(latent_all),
        "label": np.concatenate(labels),
        "predicted_label": np.concatenate(predicted_labels),
    }


def score_classifier(train_x, train_y, test_x, test_y):
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
    )
    classifier.fit(train_x, train_y)
    probability = classifier.predict_proba(test_x)[:, 1]
    prediction = probability >= 0.5
    return {
        "roc_auc": float(roc_auc_score(test_y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
        "brier_score": float(brier_score_loss(test_y, probability)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--windows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=8120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(args.checkpoint, device)
    validation = collect(model, checkpoint, args.data, "validation", args.windows, args.seed, device)
    test = collect(model, checkpoint, args.data, "test", args.windows, args.seed + 10000, device)
    feature_sets = {
        "instant_observables": ("instant",),
        "history_observables": ("history",),
        "latent_only": ("latent",),
        "instant_plus_latent": ("instant", "latent"),
        "history_plus_latent": ("history", "latent"),
    }
    classifiers = {}
    for name, fields in feature_sets.items():
        train_x = np.concatenate([validation[field] for field in fields], axis=1)
        test_x = np.concatenate([test[field] for field in fields], axis=1)
        classifiers[name] = score_classifier(
            train_x, validation["label"], test_x, test["label"]
        )
    result = {
        "checkpoint": args.checkpoint,
        "training_fields_used": ["current", "voltage", "control_mode", "event_labels"],
        "evaluation_only_gates_read": False,
        "windows_per_split": args.windows,
        "test_spike_fraction": float(test["label"].mean()),
        "model_spike_balanced_accuracy": float(
            balanced_accuracy_score(test["label"], test["predicted_label"])
        ),
        "classifiers": classifiers,
        "latent_increment_over_history_auc": float(
            classifiers["history_plus_latent"]["roc_auc"]
            - classifiers["history_observables"]["roc_auc"]
        ),
        "latent_increment_over_history_balanced_accuracy": float(
            classifiers["history_plus_latent"]["balanced_accuracy"]
            - classifiers["history_observables"]["balanced_accuracy"]
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
