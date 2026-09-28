#!/usr/bin/env python3
"""Aggregate the Latent ODE E198 same-support audit (and matched references).

Reads the per-checkpoint JSON files written by ``scripts/audit_latent_ode_chain.py``
and reports mean +/- sample SD across seeds on the identical smooth voltage-clamp
support (|dV_cmd/dt| <= 1 mV/ms), plus the plain voltage-clamp mask for reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "phase2"
OUT = RUNS / "latent_ode_e198_summary.json"
GATES = ("m", "n", "h")


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _support(payload: dict, mask: str) -> dict:
    support = payload["state_same_windows"]["paired_support"]
    if mask in support:
        return support[mask]
    # Older/published JSONs only carry the smooth block; rebuild the plain
    # voltage-clamp block from the (identical) by-mode entries.
    if mask != "voltage_clamp_all":
        raise KeyError(mask)
    by_mode = payload["state_same_windows"]["by_mode"]["voltage_clamp"]
    count = payload["transported_field_same_windows"][
        "observed_or_command_voltage_local"
    ]["voltage_clamp"]["count"]
    return {
        "definition": "all voltage-clamp samples (no smoothness restriction)",
        "count": count,
        "z_only_gate_r2": by_mode["z_only_gate_r2"],
        "voltage_only_gate_r2": by_mode["voltage_only_gate_r2"],
        "z_command_voltage_gate_r2": by_mode["z_observed_or_command_voltage_gate_r2"],
    }


def _field(payload: dict, mask: str, branch: str = "observed_or_command_voltage_local") -> dict:
    return payload["transported_field_same_windows"][branch][mask]


def row_from_payload(payload: dict) -> dict:
    smooth = _support(payload, "voltage_clamp_smooth")
    plain = _support(payload, "voltage_clamp_all")
    field_smooth = _field(payload, "voltage_clamp_smooth")
    field_plain = _field(payload, "voltage_clamp")
    field_rollout = _field(payload, "voltage_clamp_smooth", "rollout_or_command_voltage")
    row = {
        "checkpoint": payload["checkpoint"],
        "evaluation_data": payload.get("evaluation_data", payload["data"]),
        "n_points": smooth["count"],
        "n_points_voltage_clamp_plain": plain["count"],
        "frac_zero_command_slope_on_smooth_support": payload.get(
            "support_diagnostics", {}
        ).get("voltage_clamp_smooth", {}).get(
            "fraction_with_exactly_zero_command_slope", float("nan")
        ),
        "cc_response_nrmse": payload["prediction_same_windows"]["current_clamp"]["normalized_rmse"],
        "vc_response_nrmse": payload["prediction_same_windows"]["voltage_clamp"]["normalized_rmse"],
    }
    for gate in GATES:
        row[f"gate_r2_{gate}_z_command_voltage"] = smooth["z_command_voltage_gate_r2"][gate]
        row[f"gate_r2_{gate}_z_only"] = smooth["z_only_gate_r2"][gate]
        row[f"gate_r2_{gate}_voltage_only"] = smooth["voltage_only_gate_r2"][gate]
        row[f"transported_field_r2_{gate}"] = field_smooth["by_gate"][gate]["r2"]
        row[f"transported_field_relative_rmse_{gate}"] = field_smooth["by_gate"][gate]["relative_rmse"]
        row[f"transported_field_correlation_{gate}"] = field_smooth["by_gate"][gate]["correlation"]
        row[f"rollout_or_command_voltage_field_r2_{gate}"] = field_rollout["by_gate"][gate]["r2"]
        row[f"gate_r2_{gate}_z_command_voltage_voltage_clamp_plain"] = plain["z_command_voltage_gate_r2"][gate]
        row[f"gate_r2_{gate}_z_only_voltage_clamp_plain"] = plain["z_only_gate_r2"][gate]
        row[f"gate_r2_{gate}_voltage_only_voltage_clamp_plain"] = plain["voltage_only_gate_r2"][gate]
        row[f"transported_field_r2_{gate}_voltage_clamp_plain"] = field_plain["by_gate"][gate]["r2"]
        diagnostic = payload["transported_field_same_windows"].get(
            "observed_response_state_diagnostic"
        )
        if diagnostic is not None:
            row[f"transported_field_r2_{gate}_observed_response_state"] = (
                diagnostic["field_scores"]["voltage_clamp_smooth"]["by_gate"][gate]["r2"]
            )
        controls = payload["transported_field_same_windows"].get(
            "push_decomposition_controls"
        )
        if controls is not None:
            row[f"transported_field_r2_{gate}_latent_only_push"] = (
                controls["latent_only"]["voltage_clamp_smooth"]["by_gate"][gate]["r2"]
            )
            row[f"transported_field_r2_{gate}_voltage_only_push"] = (
                controls["voltage_only"]["voltage_clamp_smooth"]["by_gate"][gate]["r2"]
            )
    return row


def load_rows(pattern: str, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        path = RUNS / pattern.format(seed=seed)
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({"seed": seed, **row_from_payload(json.loads(path.read_text(encoding="utf-8")))})
    return rows


def aggregate(rows: list[dict]) -> dict:
    metrics = [key for key in rows[0] if key not in {"seed", "checkpoint", "evaluation_data"}]
    return {metric: describe([row[metric] for row in rows]) for metric in sorted(metrics)}


def section(rows: list[dict], source: str) -> dict:
    return {"source": source, "per_seed": rows, "aggregate": aggregate(rows)}


def published_structured_id_rows() -> list[dict]:
    rows = []
    for seed in range(51, 56):
        payload = json.loads(
            (RUNS / f"latentdim3_kw24_h750_s{seed}_claim_chain_e198.json").read_text(
                encoding="utf-8"
            )
        )
        rows.append({"seed": seed, **row_from_payload(payload)})
    return rows


def main() -> None:
    latent_ode_ood = load_rows("latent_ode_k2_s{seed}_claim_chain_e198.json", [56, 57, 58])
    latent_ode_id = load_rows(
        "latent_ode_k2_s{seed}_claim_chain_e198_idtest.json", [56, 57, 58]
    )
    structured_ood = load_rows(
        "latentdim3_kw24_h750_s{seed}_claim_chain_e198_ood.json", [51, 52, 53, 54, 55]
    )
    structured_id = published_structured_id_rows()

    summary = {
        "definition": (
            "Gate-decoding and transported-field metrics are evaluated on the identical "
            "held-out voltage-clamp samples satisfying |dV_cmd/dt| <= 1 mV/ms. Charts are "
            "cubic ridge (alpha=100) fitted on ID validation windows; the transported field "
            "is the central-difference pushforward of the chart along (dz/dt, dV_cmd/dt) with "
            "epsilon = 0.01 ms, compared against the canonical HH gate field."
        ),
        "protocol": {
            "windows_per_mode": 200,
            "stride": 5,
            "ridge_alpha": 100.0,
            "jvp_epsilon_ms": 0.01,
            "audit_seed": 19800,
            "mapping_split": "validation",
            "mapping_data": "data/generated/full_stratified_active_v1",
            "evaluation_split": "test",
            "id_evaluation_data": "data/generated/full_stratified_active_v1",
            "ood_evaluation_data": "data/generated/ood_protocols",
            "latent_ode_field": (
                "dz/dt = ode_func(z, response_state, u) evaluated pointwise at the free-rollout "
                "state with the true observed command u (V_cmd under voltage clamp); no integration"
            ),
            "support_finding": (
                "the voltage-clamp command is piecewise constant at the audited (0.1 ms) grid: "
                "every voltage-clamp sample has either dV_cmd/dt == 0 or |dV_cmd/dt| >= 1.62 mV/ms, "
                "so the |dV_cmd/dt| <= 1 mV/ms mask selects exactly the held-command samples "
                "(fraction with zero slope == 1.0 on every run). On that support the transported "
                "field is the pure latent pushforward of the chart, identical in definition for the "
                "structured model and the Latent ODE"
            ),
        },
        "latent_ode_k2_ood": section(
            latent_ode_ood, "runs/phase2/latent_ode_k2_s{56,57,58}_claim_chain_e198.json"
        ),
        "latent_ode_k2_id_test": section(
            latent_ode_id,
            "runs/phase2/latent_ode_k2_s{56,57,58}_claim_chain_e198_idtest.json",
        ),
        "structured_k3_ood_matched": section(
            structured_ood,
            "runs/phase2/latentdim3_kw24_h750_s{51..55}_claim_chain_e198_ood.json",
        ),
        "structured_k3_id_test_published": {
            "source": "runs/phase2/latentdim3_kw24_h750_s{51..55}_claim_chain_e198.json",
            "per_seed": structured_id,
            "aggregate": aggregate(structured_id),
        },
    }
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
