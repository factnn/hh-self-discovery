"""Directly test whether observable I/V history reconstructs learned latent state."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--windows-per-mode", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=16600)
    parser.add_argument("--max-points-per-trajectory", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--estimator", choices=("ridge", "extra_trees"), default="ridge")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(args.checkpoint, device)
    items = _collect_windows(model, checkpoint, Path(args.data), args.split, device, args.windows_per_mode, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    results = {}
    for mode in model.MODES:
        selected = [x for x in items if x["mode"] == mode]
        z = np.concatenate([x["latents"] for x in selected])
        gates = np.concatenate([x["gates"] for x in selected])
        voltage = np.concatenate([x["voltage"] for x in selected])[:, None]
        current = np.concatenate([x["current"] for x in selected])[:, None]
        current_history = np.concatenate([x["current_history"] for x in selected])
        voltage_history = np.concatenate([x["voltage_history"] for x in selected])
        encoder_summary = np.concatenate([x["encoder_history_summary"] for x in selected])
        encoder_grid = np.concatenate([x["encoder_history_grid"] for x in selected])
        groups = np.concatenate([x["trajectory_ids"] for x in selected])
        keep = []
        for trajectory_id in np.unique(groups):
            indices = np.flatnonzero(groups == trajectory_id)
            keep.extend(rng.choice(indices, min(len(indices), args.max_points_per_trajectory), replace=False))
        keep = np.asarray(keep)
        z, groups = z[keep], groups[keep]
        observable = np.column_stack([current, voltage])[keep]
        observable_dense = np.column_stack([current, voltage, current_history, voltage_history, encoder_summary, encoder_grid])[keep]
        physical = np.column_stack([gates, current, voltage])[keep]
        dense = np.column_stack([gates, current, voltage, current_history, voltage_history, encoder_summary, encoder_grid])[keep]
        fold_results = []
        for fold_index, (train, test) in enumerate(GroupKFold(5).split(physical, groups=groups)):
            predictions = {}
            for name, features in (("observable", observable), ("observable_dense_history", observable_dense), ("physical", physical), ("dense_history", dense)):
                if args.estimator == "ridge":
                    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                else:
                    reg = ExtraTreesRegressor(
                        n_estimators=args.trees,
                        max_depth=24,
                        min_samples_leaf=3,
                        max_features=0.8,
                        n_jobs=8,
                        random_state=args.seed + fold_index,
                    )
                reg.fit(features[train], z[train])
                predictions[name] = reg.predict(features[test])
            variance = float(np.sum(np.var(z[test], axis=0))) + 1e-12
            physical_mse = float(np.mean(np.sum((z[test] - predictions["physical"]) ** 2, axis=1)))
            dense_mse = float(np.mean(np.sum((z[test] - predictions["dense_history"]) ** 2, axis=1)))
            observable_mse = float(np.mean(np.sum((z[test] - predictions["observable"]) ** 2, axis=1)))
            observable_dense_mse = float(np.mean(np.sum((z[test] - predictions["observable_dense_history"]) ** 2, axis=1)))
            fold_results.append({
                "observable_r2": float(r2_score(z[test], predictions["observable"], multioutput="variance_weighted")),
                "observable_dense_history_r2": float(r2_score(z[test], predictions["observable_dense_history"], multioutput="variance_weighted")),
                "observable_residual_absorbed_fraction": 1.0 - observable_dense_mse / (observable_mse + 1e-12),
                "physical_r2": float(r2_score(z[test], predictions["physical"], multioutput="variance_weighted")),
                "dense_history_r2": float(r2_score(z[test], predictions["dense_history"], multioutput="variance_weighted")),
                "physical_normalized_mse": physical_mse / variance,
                "dense_history_normalized_mse": dense_mse / variance,
                "residual_absorbed_fraction": 1.0 - dense_mse / (physical_mse + 1e-12),
            })
        results[mode] = {
            "trajectory_count": int(len(np.unique(groups))),
            "folds": fold_results,
            "observable_r2": float(np.mean([x["observable_r2"] for x in fold_results])),
            "observable_dense_history_r2": float(np.mean([x["observable_dense_history_r2"] for x in fold_results])),
            "observable_residual_absorbed_fraction": float(np.mean([x["observable_residual_absorbed_fraction"] for x in fold_results])),
            "physical_r2": float(np.mean([x["physical_r2"] for x in fold_results])),
            "dense_history_r2": float(np.mean([x["dense_history_r2"] for x in fold_results])),
            "residual_absorbed_fraction": float(np.mean([x["residual_absorbed_fraction"] for x in fold_results])),
        }
    output = {"checkpoint": args.checkpoint, "latent_size": model.latent_size, "seed": args.seed, "estimator": args.estimator, "results": results}
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
