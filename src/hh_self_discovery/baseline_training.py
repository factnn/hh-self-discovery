"""Leak-resistant training for predictive latent baseline families."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baseline_models import MODES, build_baseline_model
from .latent_training import (
    _evaluate,
    curriculum_rollout_steps,
    multi_horizon_loss,
    post_command_transient_loss,
)
from .sampling import BalancedWindowSampler, ObservedDataset
from .training import WindowTorchDataset, fit_mode_normalizers


UPSTREAM_COMMITS = {
    "deep_delay_autoencoder": "1f43e546e9fcb0b1fc6e74ca98bc36daba98a2b6",
    "fnn": "4fdeb15639c755150dee5fda65c47f78a4e136b3",
    "latent_ode": "c0682d4f52b806fb88d965755892eadd9783f936",
    "s4": "e757cef57d89e448c413de7325ed5601aceaac13",
}


def _normalizer_payload(normalizers: dict) -> dict:
    return {
        mode: {
            "command": asdict(values.command),
            "response": asdict(values.response),
        }
        for mode, values in normalizers.items()
    }


def train_predictive_baseline(
    data_root: str,
    output_dir: str,
    model_type: str,
    latent_size: int,
    encoder_width: int = 128,
    dynamics_width: int = 128,
    epochs: int = 10,
    train_windows_per_mode: int = 4000,
    validation_windows_per_mode: int = 500,
    history_steps: int = 750,
    prediction_steps: int = 250,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-6,
    curriculum_fraction: float = 0.4,
    current_clamp_loss_weight: float = 3.0,
    voltage_transient_loss_weight: float = 1.0,
    voltage_transient_steps: int = 25,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-3,
    seed: int = 42,
    device_name: str = "cuda",
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")

    train_data = ObservedDataset(data_root, "train")
    validation_data = ObservedDataset(data_root, "validation")
    normalizers = {mode: fit_mode_normalizers(train_data, mode) for mode in MODES}
    train_samplers = {
        mode: BalancedWindowSampler(train_data, history_steps, prediction_steps)
        for mode in MODES
    }
    validation_samplers = {
        mode: BalancedWindowSampler(validation_data, history_steps, prediction_steps)
        for mode in MODES
    }
    validation_loaders = {}
    for mode in MODES:
        refs = validation_samplers[mode].sample_mode_epoch(
            mode, validation_windows_per_mode, seed + 10000
        )
        validation_loaders[mode] = DataLoader(
            WindowTorchDataset(validation_samplers[mode], refs, normalizers[mode]),
            batch_size=batch_size,
            shuffle=False,
        )

    model = build_baseline_model(
        model_type,
        latent_size,
        encoder_width=encoder_width,
        dynamics_width=dynamics_width,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "model_type": model_type,
        "latent_size": latent_size,
        "encoder_width": encoder_width,
        "dynamics_width": dynamics_width,
        "epochs": epochs,
        "train_windows_per_mode": train_windows_per_mode,
        "validation_windows_per_mode": validation_windows_per_mode,
        "history_steps": history_steps,
        "prediction_steps": prediction_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "curriculum_fraction": curriculum_fraction,
        "current_clamp_loss_weight": current_clamp_loss_weight,
        "voltage_transient_loss_weight": voltage_transient_loss_weight,
        "voltage_transient_steps": voltage_transient_steps,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "seed": seed,
        "device_name": device_name,
        "upstream_commits": UPSTREAM_COMMITS,
    }

    best_score = float("inf")
    stopping_score = float("inf")
    stale_epochs = 0
    records = []
    for epoch in range(epochs):
        rollout_steps = curriculum_rollout_steps(
            prediction_steps, epoch, epochs, curriculum_fraction
        )
        loaders = {}
        for mode in MODES:
            refs = train_samplers[mode].sample_mode_epoch(
                mode, train_windows_per_mode, seed + epoch
            )
            loaders[mode] = DataLoader(
                WindowTorchDataset(train_samplers[mode], refs, normalizers[mode]),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + epoch),
            )
        model.train()
        losses = []
        iterators = {mode: iter(loader) for mode, loader in loaders.items()}
        remaining = set(MODES)
        while remaining:
            for mode in tuple(remaining):
                try:
                    batch = next(iterators[mode])
                except StopIteration:
                    remaining.remove(mode)
                    continue
                history = batch["history"].to(device)
                command = batch["future_command"][:, :rollout_steps].to(device)
                target = batch["target"][:, :rollout_steps].to(device)
                activity = batch["activity"][:, :rollout_steps].to(device)
                prediction = model(history, command, mode)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f"non-finite {model_type} prediction")
                loss = multi_horizon_loss(prediction, target, activity)
                if mode == "voltage_clamp" and voltage_transient_loss_weight > 0.0:
                    loss = loss + voltage_transient_loss_weight * post_command_transient_loss(
                        prediction,
                        target,
                        command,
                        voltage_transient_steps,
                    )
                if mode == "current_clamp":
                    loss = current_clamp_loss_weight * loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

        metrics = _evaluate(model, validation_loaders, normalizers, device)
        score = float(np.mean([metrics[mode]["normalized_rmse"] for mode in MODES]))
        record = {
            "epoch": epoch + 1,
            "rollout_steps": rollout_steps,
            "train_loss": float(np.mean(losses)),
            "selection_score": score,
            "validation": metrics,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        payload = {
            "model_state": model.state_dict(),
            "config": config,
            "normalizers": _normalizer_payload(normalizers),
            "selection_score": score,
            "epoch": epoch + 1,
        }
        torch.save(payload, output / "last.pt")
        if score < best_score:
            best_score = score
            torch.save(payload, output / "best.pt")
        if score < stopping_score - early_stopping_min_delta:
            stopping_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1
        if early_stopping_patience and stale_epochs >= early_stopping_patience:
            break

    result = {
        "config": config,
        "best_selection_score": best_score,
        "epochs_completed": len(records),
        "history": records,
    }
    (output / "training.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
