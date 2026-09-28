"""Measure protocol-family information in latent state beyond gates and I/V."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--windows-per-mode", type=int, default=800)
    parser.add_argument("--seed", type=int, default=11200)
    parser.add_argument("--max-points-per-trajectory", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(args.checkpoint, device)
    root = Path(args.data)
    items = _collect_windows(model, checkpoint, root, args.split, device, args.windows_per_mode, args.seed)
    manifest = {r["trajectory_id"]: r for r in map(json.loads, (root / "manifest.jsonl").read_text().splitlines())}
    rng = np.random.default_rng(args.seed + 1)
    results = {}
    for mode in model.MODES:
        selected = [x for x in items if x["mode"] == mode]
        z = np.concatenate([x["latents"] for x in selected])
        gates = np.concatenate([x["gates"] for x in selected])
        voltage = np.concatenate([x["voltage"] for x in selected])[:, None]
        current = np.concatenate([x["current"] for x in selected])[:, None]
        voltage_slope = np.concatenate([x["voltage_slope"] for x in selected])[:, None]
        current_slope = np.concatenate([x["current_slope"] for x in selected])[:, None]
        voltage_history = np.concatenate([x["voltage_history"] for x in selected])
        current_history = np.concatenate([x["current_history"] for x in selected])
        encoder_history = np.concatenate([x["encoder_history_summary"] for x in selected])
        encoder_history_grid = np.concatenate([x["encoder_history_grid"] for x in selected])
        groups = np.concatenate([x["trajectory_ids"] for x in selected])
        labels = np.asarray([manifest[x]["family"] for x in groups])
        keep = []
        for trajectory_id in np.unique(groups):
            indices = np.flatnonzero(groups == trajectory_id)
            keep.extend(rng.choice(indices, min(len(indices), args.max_points_per_trajectory), replace=False))
        keep = np.asarray(keep)
        physical = np.column_stack([gates, current, voltage])[keep]
        augmented = np.column_stack([gates, current, voltage, z])[keep]
        physical_local = np.column_stack([gates, current, voltage, current_slope, voltage_slope])[keep]
        local_augmented = np.column_stack([gates, current, voltage, current_slope, voltage_slope, z])[keep]
        physical_history = np.column_stack([gates, current, voltage, current_history, voltage_history])[keep]
        history_augmented = np.column_stack([gates, current, voltage, current_history, voltage_history, z])[keep]
        physical_full_history = np.column_stack([gates, current, voltage, current_history, voltage_history, encoder_history])[keep]
        full_history_augmented = np.column_stack([gates, current, voltage, current_history, voltage_history, encoder_history, z])[keep]
        physical_dense_history = np.column_stack([gates, current, voltage, current_history, voltage_history, encoder_history, encoder_history_grid])[keep]
        dense_history_augmented = np.column_stack([gates, current, voltage, current_history, voltage_history, encoder_history, encoder_history_grid, z])[keep]
        labels, groups = labels[keep], groups[keep]
        scores = {"physical": [], "physical_plus_latent": [], "physical_local": [], "physical_local_plus_latent": [], "physical_history": [], "physical_history_plus_latent": [], "physical_full_history": [], "physical_full_history_plus_latent": [], "physical_dense_history": [], "physical_dense_history_plus_latent": []}
        for train, test in GroupKFold(5).split(physical, labels, groups):
            for name, features in (("physical", physical), ("physical_plus_latent", augmented), ("physical_local", physical_local), ("physical_local_plus_latent", local_augmented), ("physical_history", physical_history), ("physical_history_plus_latent", history_augmented), ("physical_full_history", physical_full_history), ("physical_full_history_plus_latent", full_history_augmented), ("physical_dense_history", physical_dense_history), ("physical_dense_history_plus_latent", dense_history_augmented)):
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
                clf.fit(features[train], labels[train])
                scores[name].append(balanced_accuracy_score(labels[test], clf.predict(features[test])))
        results[mode] = {
            "families": sorted(np.unique(labels).tolist()),
            "trajectory_count": int(len(np.unique(groups))),
            "physical_balanced_accuracy": float(np.mean(scores["physical"])),
            "physical_plus_latent_balanced_accuracy": float(np.mean(scores["physical_plus_latent"])),
            "latent_increment": float(np.mean(scores["physical_plus_latent"]) - np.mean(scores["physical"])),
            "physical_local_balanced_accuracy": float(np.mean(scores["physical_local"])),
            "physical_local_plus_latent_balanced_accuracy": float(np.mean(scores["physical_local_plus_latent"])),
            "latent_increment_after_local_derivatives": float(np.mean(scores["physical_local_plus_latent"]) - np.mean(scores["physical_local"])),
            "physical_history_balanced_accuracy": float(np.mean(scores["physical_history"])),
            "physical_history_plus_latent_balanced_accuracy": float(np.mean(scores["physical_history_plus_latent"])),
            "latent_increment_after_multiscale_history": float(np.mean(scores["physical_history_plus_latent"]) - np.mean(scores["physical_history"])),
            "physical_full_history_balanced_accuracy": float(np.mean(scores["physical_full_history"])),
            "physical_full_history_plus_latent_balanced_accuracy": float(np.mean(scores["physical_full_history_plus_latent"])),
            "latent_increment_after_encoder_history": float(np.mean(scores["physical_full_history_plus_latent"]) - np.mean(scores["physical_full_history"])),
            "physical_dense_history_balanced_accuracy": float(np.mean(scores["physical_dense_history"])),
            "physical_dense_history_plus_latent_balanced_accuracy": float(np.mean(scores["physical_dense_history_plus_latent"])),
            "latent_increment_after_dense_history": float(np.mean(scores["physical_dense_history_plus_latent"]) - np.mean(scores["physical_dense_history"])),
            "fold_scores": scores,
        }
    output = {"checkpoint": args.checkpoint, "latent_size": model.latent_size, "seed": args.seed, "results": results}
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
