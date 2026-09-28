"""Validate the production HH simulator against independent numerical references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hh_self_discovery.active_dataset import ACTIVE_PROTOCOLS, sample_active_protocol
from hh_self_discovery.dataset import PROTOCOL_FAMILIES, sample_protocol
from hh_self_discovery.simulator import simulate_voltage_clamp
from hh_self_discovery.simulator_validation import (
    analytic_constant_voltage,
    error_summary,
    reference_steady_state,
    solve_reference,
)
from hh_self_discovery.protocols import time_grid


FAMILIES = (*PROTOCOL_FAMILIES, *ACTIVE_PROTOCOLS)


def sample_family(family: str, dt_ms: float, seed: int):
    rng = np.random.default_rng(seed)
    if family in PROTOCOL_FAMILIES:
        return sample_protocol(family, rng, dt_ms)
    return sample_active_protocol(family, rng, dt_ms)


def compare_reference(family: str, seed: int) -> dict:
    production, metadata = sample_family(family, 0.02, seed)
    command = production.current_uA_cm2 if production.control_mode == "current_clamp" else production.voltage_mV
    reference = solve_reference(production.time_ms, command, production.control_mode)
    return {"family": family, "mode": production.control_mode, "metadata": metadata, **error_summary(production, reference)}


def convergence(family: str, seed: int) -> dict:
    traces = {dt: sample_family(family, dt, seed)[0] for dt in (0.02, 0.01, 0.005)}

    reference_errors = {}
    for dt, trace in traces.items():
        command = trace.current_uA_cm2 if trace.control_mode == "current_clamp" else trace.voltage_mV
        reference_errors[str(dt)] = error_summary(
            trace, solve_reference(trace.time_ms, command, trace.control_mode)
        )

    def compare(coarse_dt: float, fine_dt: float) -> dict[str, float]:
        coarse, fine = traces[coarse_dt], traces[fine_dt]
        gates_coarse = np.stack([coarse.n, coarse.m, coarse.h], axis=-1)
        gates_fine = np.stack(
            [np.interp(coarse.time_ms, fine.time_ms, values) for values in (fine.n, fine.m, fine.h)], axis=-1
        )
        response_coarse = coarse.voltage_mV if coarse.control_mode == "current_clamp" else coarse.current_uA_cm2
        response_fine_raw = fine.voltage_mV if fine.control_mode == "current_clamp" else fine.current_uA_cm2
        response_fine = np.interp(coarse.time_ms, fine.time_ms, response_fine_raw)
        return {
            "response_rmse": float(np.sqrt(np.mean((response_coarse - response_fine) ** 2))),
            "response_max_abs": float(np.max(np.abs(response_coarse - response_fine))),
            "gate_rmse": float(np.sqrt(np.mean((gates_coarse - gates_fine) ** 2))),
            "gate_max_abs": float(np.max(np.abs(gates_coarse - gates_fine))),
        }

    return {
        "family": family,
        "mode": traces[0.02].control_mode,
        "production_vs_reference_by_dt": reference_errors,
        "dt_0.02_vs_0.01": compare(0.02, 0.01),
        "dt_0.01_vs_0.005": compare(0.01, 0.005),
    }


def analytic_checks() -> list[dict]:
    results = []
    for voltage in (-100.0, -65.0, -20.0, 40.0):
        time = time_grid(40.0, 0.02)
        initial = reference_steady_state(-65.0)
        production = simulate_voltage_clamp(
            time, np.full_like(time, voltage), initial_gates=initial, include_capacitive_current=False
        )
        expected = analytic_constant_voltage(time, voltage, initial)
        actual = np.stack([production.n, production.m, production.h], axis=-1)
        results.append(
            {
                "voltage_mV": voltage,
                "gate_rmse": float(np.sqrt(np.mean((actual - expected) ** 2))),
                "gate_max_abs": float(np.max(np.abs(actual - expected))),
            }
        )
    return results


def plot_summary(report: dict, output: Path) -> None:
    labels = [item["family"].replace("current_", "I:").replace("voltage_", "V:").replace("active_", "A:") for item in report["independent_reference"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    axes[0].bar(x, [item["gate_rmse"] for item in report["independent_reference"]])
    axes[0].set(yscale="log", ylabel="Gate RMSE", title="Production vs DOP853")
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].bar(x - 0.18, [item["dt_0.02_vs_0.01"]["response_rmse"] for item in report["convergence"]], width=0.36, label=".02 vs .01")
    axes[1].bar(x + 0.18, [item["dt_0.01_vs_0.005"]["response_rmse"] for item in report["convergence"]], width=0.36, label=".01 vs .005")
    axes[1].set(yscale="log", ylabel="Response RMSE", title="Step-size convergence")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].legend(frameon=False, fontsize=8)
    voltages = [str(int(item["voltage_mV"])) for item in report["analytic_voltage_clamp"]]
    axes[2].bar(voltages, [item["gate_max_abs"] for item in report["analytic_voltage_clamp"]])
    axes[2].set(yscale="log", xlabel="Clamp voltage (mV)", ylabel="Max gate error", title="Constant-voltage analytic check")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/simulator_validation")
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": args.seed,
        "families": list(FAMILIES),
        "independent_reference": [compare_reference(family, args.seed + index) for index, family in enumerate(FAMILIES)],
        "convergence": [convergence(family, args.seed + index) for index, family in enumerate(FAMILIES)],
        "analytic_voltage_clamp": analytic_checks(),
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    plot_summary(report, output / "validation.png")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
