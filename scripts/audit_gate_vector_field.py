"""Audit whether a blind latent-to-gate map preserves the HH vector field."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint
from hh_self_discovery.simulator import gate_derivatives


def field_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    scale = np.sqrt(np.mean(target**2)) + 1e-12
    cosine = np.sum(predicted * target, axis=1) / (
        np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1) + 1e-12
    )
    return {
        "relative_rmse": float(np.sqrt(np.mean((predicted - target) ** 2)) / scale),
        "mean_cosine": float(np.mean(cosine)),
        "median_cosine": float(np.median(cosine)),
        "positive_direction_fraction": float(np.mean(cosine > 0)),
    }


def stratified_metrics(predicted: np.ndarray, target: np.ndarray, gates: np.ndarray) -> dict:
    gate_names = ("n", "m", "h")
    by_gate = {}
    for index, name in enumerate(gate_names):
        scale = np.sqrt(np.mean(target[:, index] ** 2)) + 1e-12
        by_gate[name] = {
            "relative_rmse": float(
                np.sqrt(np.mean((predicted[:, index] - target[:, index]) ** 2)) / scale
            ),
            "pearson": float(np.corrcoef(predicted[:, index], target[:, index])[0, 1]),
            "sign_agreement": float(
                np.mean(np.sign(predicted[:, index]) == np.sign(target[:, index]))
            ),
        }
    speed = np.linalg.norm(target, axis=1)
    edges = np.quantile(speed, [0.0, 0.25, 0.5, 0.75, 1.0])
    by_speed = {}
    for index in range(4):
        upper = speed <= edges[index + 1] if index == 3 else speed < edges[index + 1]
        mask = (speed >= edges[index]) & upper
        by_speed[f"q{index + 1}"] = {
            "speed_range": [float(edges[index]), float(edges[index + 1])],
            "count": int(mask.sum()),
            **field_metrics(predicted[mask], target[mask]),
        }
    valid = np.all((gates >= 0.0) & (gates <= 1.0), axis=1)
    return {
        "by_gate": by_gate,
        "by_true_speed_quartile": by_speed,
        "physical_gate_subset": {
            "count": int(valid.sum()),
            **field_metrics(predicted[valid], target[valid]),
        },
    }


def evaluate(checkpoint_path: str, data_root: str, device_name: str, seed: int) -> dict:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    root = Path(data_root)
    mapping_items = _collect_windows(
        model, checkpoint, root, "validation", device, 300, seed
    )
    test_items = _collect_windows(
        model, checkpoint, root, "test", device, 300, seed + 10000
    )
    mapping_z = np.concatenate([item["latents"] for item in mapping_items])
    mapping_g = np.concatenate([item["gates"] for item in mapping_items])
    readout = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=1e-3),
    ).fit(mapping_z, mapping_g)
    z = np.concatenate([item["latents"] for item in test_items])
    voltage = np.concatenate([item["voltage"] for item in test_items])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(z), size=min(30000, len(z)), replace=False)
    z, voltage = z[chosen], voltage[chosen]
    z_scale = mapping_z.std(axis=0) + 1e-8
    output = {}
    for radius in (0.0, 0.05, 0.1, 0.2):
        direction = rng.normal(size=z.shape)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
        state = np.clip(z + radius * direction * z_scale, 1e-5, 1.0 - 1e-5)
        voltage_tensor = torch.as_tensor(voltage, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            steady, tau = model.gate_kinetics(voltage_tensor)
            latent_field = ((steady - state_tensor) / tau).cpu().numpy()
        gates = readout.predict(state)
        true_field = np.stack(
            [gate_derivatives(v, g) for v, g in zip(voltage, gates)]
        )
        epsilon = 1e-4
        pushed = (
            readout.predict(state + epsilon * latent_field)
            - readout.predict(state - epsilon * latent_field)
        ) / (2.0 * epsilon)
        output[f"radius_{radius:.2f}"] = {
            **field_metrics(pushed, true_field),
            "gate_out_of_bounds_fraction": float(np.mean((gates < 0) | (gates > 1))),
            **stratified_metrics(pushed, true_field, gates),
        }
    return {
        "checkpoint": checkpoint_path,
        "data_root": data_root,
        "seed": seed,
        "samples": int(len(z)),
        "perturbation_units": "per-coordinate validation latent standard deviation",
        "radii": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=4646)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data, args.device, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
