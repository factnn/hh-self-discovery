#!/usr/bin/env python3
"""Support composition of the shared audit windows, and the chart-objective sweep.

Two descriptive aggregates that the appendix reports:

* how the shared held-out windows are drawn, and how many independent
  trajectories the smooth voltage-clamp support actually draws on;
* the state--field frontier of the dynamics-aware chart objective, over every
  field weight and chart initialization present in ``runs/phase2``.

    python scripts/audit_chart_objective.py --checkpoint <structured ckpt> \
        --data data/generated/full_stratified_active_v1 --output <out.json>
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint
from hh_self_discovery.sampling import ObservedDataset

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "phase2"


def support_composition(checkpoint_path: str, data: str, device_name: str,
                        windows_per_mode: int, stride: int, seed: int) -> dict:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    checkpoint = dict(checkpoint)
    checkpoint["_audit_stride"] = stride
    root = Path(data)
    dataset = ObservedDataset(root, "test")
    records = {record["trajectory_id"]: record for record in dataset.records}
    items = _collect_windows(
        model, checkpoint, root, "test", device, windows_per_mode, seed, stride=stride,
        strict_sampling=True,
    )
    steps = len(range(0, checkpoint["prediction_steps"], stride))
    report = {}
    for mode in ("voltage_clamp", "current_clamp"):
        subset = [item for item in items if item["mode"] == mode]
        windows = []
        smooth_pairs = 0
        for item in subset:
            ids = np.asarray(item["trajectory_ids"]).reshape(-1, steps)
            windows.extend(ids[:, 0].tolist())
            slope = item["voltage_slope"].reshape(-1, steps)
            if mode == "voltage_clamp":
                smooth_pairs += int(
                    ((np.abs(slope[:, :-1]) <= 1.0) & (np.abs(slope[:, 1:]) <= 1.0)).sum()
                )
        report[mode] = {
            "windows": len(windows),
            "distinct_trajectories_sampled": len(set(windows)),
            "test_trajectories_available": sum(
                1 for r in dataset.records
                if r["control_mode"] == mode and r["split"] == "test"
            ),
            "family_window_counts": dict(sorted(Counter(
                records[t]["family"] for t in windows).items())),
            "smooth_support_pairs": smooth_pairs,
        }
    return {
        "definition": (
            "Windows are drawn by selecting held-out test trajectories uniformly and "
            "then a window uniformly within each; scores pool points, so every window "
            "carries equal weight regardless of trajectory length."
        ),
        "stride": stride,
        "windows_per_mode": windows_per_mode,
        "seed": seed,
        "by_mode": report,
    }


def chart_objective(field_weights=("0", "0.01", "0.1", "1", "10")) -> dict:
    files = sorted(glob.glob(str(RUNS / "*dynamics_aware_map*.json")))
    per: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for path in files:
        payload = json.load(open(path))
        checkpoint = Path(payload["checkpoint"]).parent.name
        for key, block in payload["results"].items():
            weight = key.replace("field_weight_", "")
            test = block["test"]
            per[checkpoint][weight].append({
                "gate_r2": test["mean_gate_r2"],
                "field_relative_rmse": test["field_relative_rmse"],
                "field_cosine": test["field_mean_cosine"],
            })
    rows = []
    for checkpoint in sorted(per):
        for weight in field_weights:
            if weight not in per[checkpoint]:
                continue
            runs = per[checkpoint][weight]
            rows.append({
                "checkpoint": checkpoint,
                "field_weight": float(weight),
                "runs": len(runs),
                "gate_r2_mean": st.mean([r["gate_r2"] for r in runs]),
                "gate_r2_min": min(r["gate_r2"] for r in runs),
                "gate_r2_max": max(r["gate_r2"] for r in runs),
                "field_relative_rmse_mean": st.mean([r["field_relative_rmse"] for r in runs]),
                "field_relative_rmse_min": min(r["field_relative_rmse"] for r in runs),
                "field_relative_rmse_max": max(r["field_relative_rmse"] for r in runs),
            })
    return {
        "definition": (
            "Dynamics-aware cubic chart on the same validation split and evaluation "
            "windows, sweeping the weight on the transported-field term."
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="structured checkpoint, used only to reproduce the window sampler")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--windows-per-mode", type=int, default=200)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=29800)
    args = parser.parse_args()
    payload = {
        "support_composition": support_composition(
            args.checkpoint, args.data, args.device, args.windows_per_mode, args.stride, args.seed
        ),
        "chart_objective_sweep": chart_objective(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
