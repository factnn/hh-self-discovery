"""Post-selection comparison of learned and canonical HH kinetics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .blind_evaluation import (
    TRUE_GATES,
    _collect_windows,
    _records,
    _rollout_record,
    load_latent_checkpoint,
)
from .simulator import gate_rates


def _true_kinetics(voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steady, tau = [], []
    for value in voltage:
        an, bn, am, bm, ah, bh = gate_rates(float(value))
        steady.append([an / (an + bn), am / (am + bm), ah / (ah + bh)])
        tau.append([1.0 / (an + bn), 1.0 / (am + bm), 1.0 / (ah + bh)])
    return np.asarray(steady), np.asarray(tau)


def evaluate_mechanism(
    checkpoint_path: str | Path,
    blind_result_path: str | Path,
    output_figure: str | Path,
) -> dict:
    model, _ = load_latent_checkpoint(checkpoint_path, torch.device("cpu"))
    blind = json.loads(Path(blind_result_path).read_text(encoding="utf-8"))
    assignments = blind["best_monotonic_assignment"]
    voltage = np.linspace(-100.0, 50.0, 301)
    with torch.no_grad():
        learned_steady, learned_tau = model.gate_kinetics(
            torch.from_numpy(voltage.astype(np.float32))
        )
    learned_steady = learned_steady.numpy()
    learned_tau = learned_tau.numpy()
    true_steady, true_tau = _true_kinetics(voltage)
    gate_index = {name: index for index, name in enumerate(TRUE_GATES)}
    comparisons = []
    for assignment in assignments:
        latent = assignment["latent"]
        gate = assignment["gate"]
        true_index = gate_index[gate]
        comparisons.append({
            "latent": latent,
            "gate": gate,
            "steady_spearman": float(
                spearmanr(learned_steady[:, latent], true_steady[:, true_index]).statistic
            ),
            "tau_log_spearman": float(
                spearmanr(np.log(learned_tau[:, latent]), np.log(true_tau[:, true_index])).statistic
            ),
            "tau_log_rmse": float(
                np.sqrt(np.mean((np.log(learned_tau[:, latent]) - np.log(true_tau[:, true_index])) ** 2))
            ),
        })

    figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for column, assignment in enumerate(assignments):
        latent, gate = assignment["latent"], assignment["gate"]
        true_index = gate_index[gate]
        axes[0, column].plot(voltage, true_steady[:, true_index], label=f"true {gate}")
        axes[0, column].plot(voltage, learned_steady[:, latent], label=f"z{latent}")
        axes[1, column].semilogy(voltage, true_tau[:, true_index], label=f"true {gate}")
        axes[1, column].semilogy(voltage, learned_tau[:, latent], label=f"z{latent}")
        axes[0, column].legend()
        axes[1, column].legend()
        axes[1, column].set_xlabel("Voltage (mV)")
    axes[0, 0].set_ylabel("Steady state")
    axes[1, 0].set_ylabel("Time constant (ms)")
    figure.tight_layout()
    output = Path(output_figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return {
        "checkpoint": str(checkpoint_path),
        "blind_result": str(blind_result_path),
        "figure": str(output),
        "voltage_range_mV": [-100.0, 50.0],
        "comparisons": comparisons,
    }


def evaluate_phase_portrait(
    checkpoint_path: str | Path,
    data_root: str | Path,
    output_figure: str | Path,
    windows_per_mode: int = 500,
    seed: int = 6060,
) -> dict:
    """Compare mapped latent and true gate phase portraits on one held-out trace."""
    device = torch.device("cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    root = Path(data_root)
    mapping = _collect_windows(
        model, checkpoint, root, "validation", device, windows_per_mode, seed
    )
    mapping_z = np.concatenate([item["latents"] for item in mapping])
    mapping_gates = np.concatenate([item["gates"] for item in mapping])
    readout = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=1e-3),
    )
    readout.fit(mapping_z, mapping_gates)
    candidates = [
        record for record in _records(root, "test")
        if record["control_mode"] == "current_clamp"
    ]
    activity_means = []
    for record in candidates:
        with np.load(root / record["observed_path"]) as archive:
            activity_means.append(float(archive["activity_score"].mean()))
    record = candidates[int(np.argmax(activity_means))]
    rollout = _rollout_record(model, checkpoint, root, record, device, stride=5)
    mapped_gates = readout.predict(rollout["latents"])
    voltage = rollout["target"]
    dt_ms = checkpoint["dt_ms"] * 5
    figure, axes = plt.subplots(2, 3, figsize=(12, 6))
    for index, gate in enumerate(TRUE_GATES):
        axes[0, index].plot(voltage, rollout["gates"][:, index], ".", ms=1.5, label=f"true {gate}")
        axes[0, index].plot(voltage, mapped_gates[:, index], ".", ms=1.5, label="mapped z")
        axes[0, index].set_xlabel("Voltage (mV)")
        axes[0, index].set_ylabel(gate)
        axes[0, index].legend()
        true_derivative = np.gradient(rollout["gates"][:, index], dt_ms)
        mapped_derivative = np.gradient(mapped_gates[:, index], dt_ms)
        axes[1, index].plot(
            rollout["gates"][:, index], true_derivative, ".", ms=1.5, label="true"
        )
        axes[1, index].plot(
            mapped_gates[:, index], mapped_derivative, ".", ms=1.5, label="mapped z"
        )
        axes[1, index].set_xlabel(gate)
        axes[1, index].set_ylabel(f"d{gate}/dt")
        axes[1, index].legend()
    figure.tight_layout()
    output = Path(output_figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return {
        "trajectory_id": record["trajectory_id"],
        "control_mode": record["control_mode"],
        "mean_activity": max(activity_means),
        "figure": str(output),
    }
