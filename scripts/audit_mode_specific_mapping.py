"""Test whether clamp modes require separate latent-to-gate coordinate charts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint
from hh_self_discovery.simulator import gate_derivatives


def fit_map(items):
    z = np.concatenate([item["latents"] for item in items])
    gates = np.concatenate([item["gates"] for item in items])
    return make_pipeline(StandardScaler(), Ridge(alpha=0.1)).fit(z, gates)


def evaluate_map(model, mapping, items, device):
    z = np.concatenate([item["latents"] for item in items])
    gates = np.concatenate([item["gates"] for item in items])
    voltage = np.concatenate([item["voltage"] for item in items])
    mapped = mapping.predict(z)
    with torch.no_grad():
        voltage_tensor = torch.as_tensor(voltage, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(z, dtype=torch.float32, device=device)
        steady, tau = model.gate_kinetics(voltage_tensor)
        latent_field = ((steady - state_tensor) / tau).cpu().numpy()
    epsilon = 1e-4
    pushed = (
        mapping.predict(z + epsilon * latent_field)
        - mapping.predict(z - epsilon * latent_field)
    ) / (2 * epsilon)
    true_field = np.stack([gate_derivatives(v, g) for v, g in zip(voltage, mapped)])
    scale = np.sqrt(np.mean(true_field**2)) + 1e-12
    cosine = np.sum(pushed * true_field, axis=1) / (
        np.linalg.norm(pushed, axis=1) * np.linalg.norm(true_field, axis=1) + 1e-12
    )
    return {
        "mean_gate_r2": float(r2_score(gates, mapped, multioutput="variance_weighted")),
        "gate_r2": {
            gate: float(r2_score(gates[:, i], mapped[:, i]))
            for i, gate in enumerate(("n", "m", "h"))
        },
        "vector_field_relative_rmse": float(
            np.sqrt(np.mean((pushed - true_field) ** 2)) / scale
        ),
        "vector_field_mean_cosine": float(np.mean(cosine)),
        "vector_field_positive_fraction": float(np.mean(cosine > 0)),
    }


def evaluate(checkpoint_path, data_root, device_name, seed):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    root = Path(data_root)
    validation = _collect_windows(model, checkpoint, root, "validation", device, 300, seed)
    test = _collect_windows(model, checkpoint, root, "test", device, 300, seed + 10000)
    joint = fit_map(validation)
    output = {}
    for mode in model.MODES:
        validation_mode = [item for item in validation if item["mode"] == mode]
        test_mode = [item for item in test if item["mode"] == mode]
        output[mode] = {
            "joint_map": evaluate_map(model, joint, test_mode, device),
            "mode_specific_map": evaluate_map(
                model, fit_map(validation_mode), test_mode, device
            ),
        }
    return {"checkpoint": checkpoint_path, "mapping": "linear_ridge_0.1", "results": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=5454)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data, args.device, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
