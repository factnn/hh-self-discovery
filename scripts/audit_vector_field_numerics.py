"""Positive controls for analytic versus finite-difference gate velocities."""

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


def compare(predicted: np.ndarray, target: np.ndarray) -> dict:
    scale = np.sqrt(np.mean(target**2)) + 1e-12
    cosine = np.sum(predicted * target, axis=1) / (
        np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1) + 1e-12
    )
    return {
        "relative_rmse": float(np.sqrt(np.mean((predicted - target) ** 2)) / scale),
        "mean_cosine": float(np.mean(cosine)),
        "median_cosine": float(np.median(cosine)),
    }


def evaluate(checkpoint_path: str, data_root: str, device_name: str, seed: int) -> dict:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    root = Path(data_root)
    validation = _collect_windows(model, checkpoint, root, "validation", device, 300, seed)
    test = _collect_windows(model, checkpoint, root, "test", device, 300, seed + 10000)
    train_z = np.concatenate([item["latents"] for item in validation])
    train_g = np.concatenate([item["gates"] for item in validation])
    mapping = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=1, include_bias=False),
        Ridge(alpha=0.1),
    ).fit(train_z, train_g)
    dt_ms = checkpoint["dt_ms"] * 5
    steps = checkpoint["prediction_steps"] // 5
    outputs = {}
    for mode in model.MODES:
        items = [item for item in test if item["mode"] == mode]
        z = np.concatenate([item["latents"] for item in items]).reshape(-1, steps, model.latent_size)
        gates = np.concatenate([item["gates"] for item in items]).reshape(-1, steps, 3)
        true_v = np.concatenate([item["voltage"] for item in items]).reshape(-1, steps)
        model_v = np.concatenate([item["model_voltage"] for item in items]).reshape(-1, steps)
        mapped = mapping.predict(z.reshape(-1, model.latent_size)).reshape(-1, steps, 3)
        mapped_fd = np.gradient(mapped, dt_ms, axis=1)
        true_fd = np.gradient(gates, dt_ms, axis=1)
        flat_z = z.reshape(-1, model.latent_size)
        flat_model_v = model_v.reshape(-1)
        with torch.no_grad():
            voltage_tensor = torch.as_tensor(flat_model_v, dtype=torch.float32, device=device)
            state_tensor = torch.as_tensor(flat_z, dtype=torch.float32, device=device)
            steady, tau = model.gate_kinetics(voltage_tensor)
            latent_field = ((steady - state_tensor) / tau).cpu().numpy()
        epsilon = 1e-4
        mapped_analytic = (
            mapping.predict(flat_z + epsilon * latent_field)
            - mapping.predict(flat_z - epsilon * latent_field)
        ) / (2.0 * epsilon)
        true_analytic = np.stack(
            [
                gate_derivatives(v, g)
                for v, g in zip(true_v.reshape(-1), gates.reshape(-1, 3))
            ]
        )
        trim = np.ones((len(z), steps), dtype=bool)
        trim[:, :2] = False
        trim[:, -2:] = False
        mask = trim.reshape(-1)
        outputs[mode] = {
            "learned_generator_vs_mapped_trajectory_fd": compare(
                mapped_analytic[mask], mapped_fd.reshape(-1, 3)[mask]
            ),
            "hh_generator_vs_true_trajectory_fd": compare(
                true_analytic[mask], true_fd.reshape(-1, 3)[mask]
            ),
        }
    return {"checkpoint": checkpoint_path, "dt_ms": dt_ms, "results": outputs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=5252)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data, args.device, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
