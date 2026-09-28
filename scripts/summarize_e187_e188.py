#!/usr/bin/env python3
"""Summarize the completed event-weight factor audit without modifying results."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "phase2"
BINS = (
    "jump_distance_le_0p5ms",
    "jump_distance_0p5_to_2ms",
    "jump_distance_gt_2ms",
)


def load(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def matching(patterns: list[str]) -> list[str]:
    return sorted({path for pattern in patterns for path in glob.glob(str(RUNS / pattern))})


def event_summary(branch: int, weight: int) -> None:
    if weight == 1:
        patterns = [f"branch{branch}_transientw1*_eventbins_e18*.json"]
    else:
        patterns = [
            f"branch{branch}_transientw2*_eventbins_e182*.json",
            f"branch{branch}_transientw2*_eventdecomp_e187*.json",
            f"branch{branch}_transientw2*_eventdecomp_e188*.json",
        ]
    rows = []
    for path in matching(patterns):
        metrics = load(path)["results"]["field_weight_1"]["test"]["stratified"]["voltage_clamp"]
        row = []
        for name in BINS:
            item = metrics[name]
            row.extend(
                (
                    item["per_gate_field"]["m"]["relative_rmse"],
                    item["per_gate_field"]["m"]["correlation"],
                    item["gate_r2"]["m"],
                )
            )
        rows.append(row)
    values = np.asarray(rows)
    print(f"branch={branch} weight={weight} repeats={len(rows)}")
    for index, name in enumerate(BINS):
        print(
            f"  {name}: rmse={values[:, 3*index].mean():.4f}+/-{values[:, 3*index].std():.4f} "
            f"corr={values[:, 3*index+1].mean():.4f}+/-{values[:, 3*index+1].std():.4f} "
            f"state_r2={values[:, 3*index+2].mean():.6f}"
        )


def decomposition_summary(branch: int) -> None:
    paths = matching(
        [
            f"branch{branch}_transientw2*_eventdecomp_e187*.json",
            f"branch{branch}_transientw2*_eventdecomp_e188*.json",
        ]
    )
    rows = []
    for path in paths:
        item = load(path)["results"]["field_weight_1"]["test"]["stratified"]["voltage_clamp"][BINS[-1]]
        terms = item["field_error_decomposition"]["m"]
        rows.append(
            (
                terms["state_amplification_rmse"],
                terms["coordinate_residual_rmse"],
                terms["cross_term"],
                terms["error_correlation"],
                terms["total_rmse"],
                item["gate_r2"]["m"],
            )
        )
    values = np.asarray(rows)
    print(f"branch={branch} weight=2 far-event decomposition repeats={len(rows)}")
    for name, column in zip(
        ("state_amp", "coord_residual", "cross", "error_corr", "total", "state_r2"),
        values.T,
    ):
        print(
            f"  {name}: {column.mean():.6f}+/-{column.std():.6f} "
            f"range={column.min():.6f}..{column.max():.6f}"
        )


def show_e188_physical() -> None:
    print("E188 physical JSON numeric leaves")
    for path in matching(["branch2_transientw2*_e188*.json"]):
        if "eventdecomp" in path:
            continue
        print(Path(path).name)

        def walk(value: object, prefix: str = "") -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(child, f"{prefix}.{key}" if prefix else key)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"  {prefix}={value}")

        walk(load(path))


if __name__ == "__main__":
    for branch_id in (2, 5):
        for event_weight in (1, 2):
            event_summary(branch_id, event_weight)
        decomposition_summary(branch_id)
    show_e188_physical()
