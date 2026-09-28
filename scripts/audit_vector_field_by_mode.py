"""Compare mapped and HH vector fields separately by clamp mode."""

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


def metrics(predicted, target, mask):
    predicted, target = predicted[mask], target[mask]
    scale = np.sqrt(np.mean(target**2)) + 1e-12
    cosine = np.sum(predicted * target, axis=1) / (
        np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1) + 1e-12
    )
    return {
        "count": int(mask.sum()),
        "relative_rmse": float(np.sqrt(np.mean((predicted - target) ** 2)) / scale),
        "mean_cosine": float(np.mean(cosine)),
        "median_cosine": float(np.median(cosine)),
        "positive_fraction": float(np.mean(cosine > 0)),
    }


def evaluate(checkpoint_path, data_root, device_name, seed):
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
    epsilon = 1e-4
    steps = checkpoint["prediction_steps"] // 5
    dt_ms = checkpoint["dt_ms"] * 5
    output = {}
    for mode in model.MODES:
        items = [item for item in test if item["mode"] == mode]
        z = np.concatenate([item["latents"] for item in items])
        true_v = np.concatenate([item["voltage"] for item in items])
        model_v = np.concatenate([item["model_voltage"] for item in items])
        mapped = mapping.predict(z)
        true_field = np.stack([gate_derivatives(v, g) for v, g in zip(true_v, mapped)])
        fields = {}
        for name, voltage in (("common_true_voltage", true_v), ("trajectory_voltage", model_v)):
            with torch.no_grad():
                voltage_tensor = torch.as_tensor(voltage, dtype=torch.float32, device=device)
                state_tensor = torch.as_tensor(z, dtype=torch.float32, device=device)
                steady, tau = model.gate_kinetics(voltage_tensor)
                latent_field = ((steady - state_tensor) / tau).cpu().numpy()
            fields[name] = (
                mapping.predict(z + epsilon * latent_field)
                - mapping.predict(z - epsilon * latent_field)
            ) / (2 * epsilon)
        all_mask = np.ones(len(z), dtype=bool)
        masks = {"all": all_mask}
        if mode == "voltage_clamp":
            voltage_series = true_v.reshape(-1, steps)
            dvdt = np.abs(np.gradient(voltage_series, dt_ms, axis=1)).reshape(-1)
            masks["smooth_dvdt_le_5"] = dvdt <= 5.0
            masks["smooth_dvdt_le_1"] = dvdt <= 1.0
        output[mode] = {
            field_name: {
                mask_name: metrics(field, true_field, mask)
                for mask_name, mask in masks.items()
            }
            for field_name, field in fields.items()
        }
    return {"checkpoint": checkpoint_path, "mapping": "linear_ridge_0.1", "results": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=5353)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data, args.device, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
