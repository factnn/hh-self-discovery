#!/usr/bin/env python3
"""Aggregate E198 same-support state and transported-field audits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "phase2"
OUT = RUNS / "e198_same_support_summary.json"


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    records: dict[int, list[dict]] = {3: [], 5: [], 6: []}
    for latent_dim in records:
        for seed in range(51, 56):
            path = RUNS / (
                f"latentdim{latent_dim}_kw24_h750_s{seed}_claim_chain_e198.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            prediction = payload["prediction_same_windows"]
            cc_state = payload["state_same_windows"]["by_mode"][
                "current_clamp"
            ]
            support = payload["state_same_windows"]["paired_support"][
                "voltage_clamp_smooth"
            ]
            field = payload["transported_field_same_windows"][
                "observed_or_command_voltage_local"
            ]["voltage_clamp_smooth"]["by_gate"]["m"]
            records[latent_dim].append(
                {
                    "seed": seed,
                    "count": support["count"],
                    "cc_response_nrmse": prediction["current_clamp"]["normalized_rmse"],
                    "vc_response_nrmse": prediction["voltage_clamp"]["normalized_rmse"],
                    "cc_m_state_z_only_r2": cc_state["z_only_gate_r2"]["m"],
                    "cc_m_state_z_observed_voltage_r2": cc_state[
                        "z_observed_or_command_voltage_gate_r2"
                    ]["m"],
                    "cc_m_state_z_rollout_voltage_r2": cc_state[
                        "z_rollout_or_command_voltage_gate_r2"
                    ]["m"],
                    "m_state_z_only_r2": support["z_only_gate_r2"]["m"],
                    "m_state_voltage_only_r2": support["voltage_only_gate_r2"]["m"],
                    "m_state_z_command_voltage_r2": support[
                        "z_command_voltage_gate_r2"
                    ]["m"],
                    "m_field_r2": field["r2"],
                    "m_field_relative_rmse": field["relative_rmse"],
                }
            )

    metrics = (
        "cc_response_nrmse",
        "vc_response_nrmse",
        "cc_m_state_z_only_r2",
        "cc_m_state_z_observed_voltage_r2",
        "cc_m_state_z_rollout_voltage_r2",
        "m_state_z_only_r2",
        "m_state_voltage_only_r2",
        "m_state_z_command_voltage_r2",
        "m_field_r2",
        "m_field_relative_rmse",
    )
    summary = {
        "definition": (
            "State and transported-field metrics use the identical held-out "
            "voltage-clamp samples satisfying |dV_cmd/dt| <= 1 mV/ms."
        ),
        "by_latent_dimension": {
            str(latent_dim): {
                "per_seed": rows,
                "aggregate": {
                    metric: describe([row[metric] for row in rows])
                    for metric in metrics
                },
            }
            for latent_dim, rows in records.items()
        },
    }

    paired = []
    for row5, row6 in zip(records[5], records[6], strict=True):
        paired.append(
            {
                "seed": row5["seed"],
                "delta_state_z_only_k6_minus_k5": (
                    row6["m_state_z_only_r2"] - row5["m_state_z_only_r2"]
                ),
                "delta_state_zv_k6_minus_k5": (
                    row6["m_state_z_command_voltage_r2"]
                    - row5["m_state_z_command_voltage_r2"]
                ),
                "delta_field_k6_minus_k5": (
                    row6["m_field_r2"] - row5["m_field_r2"]
                ),
            }
        )
    summary["paired_k6_minus_k5"] = {
        "per_seed": paired,
        "aggregate": {
            metric: describe([row[metric] for row in paired])
            for metric in (
                "delta_state_z_only_k6_minus_k5",
                "delta_state_zv_k6_minus_k5",
                "delta_field_k6_minus_k5",
            )
        },
        "field_declines": int(sum(row["delta_field_k6_minus_k5"] < 0 for row in paired)),
    }
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
