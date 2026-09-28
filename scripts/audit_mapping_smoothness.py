"""Sweep blind-map complexity to test vector-field conclusions for robustness."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint
from hh_self_discovery.simulator import gate_derivatives


def evaluate(checkpoint_path: str, data_root: str, device_name: str, seed: int) -> dict:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    root = Path(data_root)
    validation = _collect_windows(model, checkpoint, root, "validation", device, 300, seed)
    test = _collect_windows(model, checkpoint, root, "test", device, 300, seed + 10000)
    train_z = np.concatenate([item["latents"] for item in validation])
    train_g = np.concatenate([item["gates"] for item in validation])
    test_z = np.concatenate([item["latents"] for item in test])
    test_g = np.concatenate([item["gates"] for item in test])
    voltage = np.concatenate([item["voltage"] for item in test])
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(test_z), size=min(30000, len(test_z)), replace=False)
    test_z, test_g, voltage = test_z[selected], test_g[selected], voltage[selected]
    with torch.no_grad():
        voltage_tensor = torch.as_tensor(voltage, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(test_z, dtype=torch.float32, device=device)
        steady, tau = model.gate_kinetics(voltage_tensor)
        latent_field = ((steady - state_tensor) / tau).cpu().numpy()
    results = {}
    for degree in (1, 2, 3):
        for alpha in (1e-1, 1e-2, 1e-3, 1e-4):
            name = f"degree{degree}_alpha{alpha:.0e}"
            mapping = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=degree, include_bias=False),
                Ridge(alpha=alpha),
            ).fit(train_z, train_g)
            mapped = mapping.predict(test_z)
            epsilon = 1e-4
            pushed = (
                mapping.predict(test_z + epsilon * latent_field)
                - mapping.predict(test_z - epsilon * latent_field)
            ) / (2.0 * epsilon)
            true_field = np.stack(
                [gate_derivatives(v, g) for v, g in zip(voltage, mapped)]
            )
            field_scale = np.sqrt(np.mean(true_field**2)) + 1e-12
            cosine = np.sum(pushed * true_field, axis=1) / (
                np.linalg.norm(pushed, axis=1) * np.linalg.norm(true_field, axis=1) + 1e-12
            )
            results[name] = {
                "degree": degree,
                "alpha": alpha,
                "mean_gate_r2": float(
                    r2_score(test_g, mapped, multioutput="variance_weighted")
                ),
                "gate_r2": {
                    gate: float(r2_score(test_g[:, index], mapped[:, index]))
                    for index, gate in enumerate(("n", "m", "h"))
                },
                "gate_out_of_bounds_fraction": float(
                    np.mean((mapped < 0.0) | (mapped > 1.0))
                ),
                "vector_field_relative_rmse": float(
                    np.sqrt(np.mean((pushed - true_field) ** 2)) / field_scale
                ),
                "vector_field_mean_cosine": float(np.mean(cosine)),
                "vector_field_positive_fraction": float(np.mean(cosine > 0.0)),
            }
    return {
        "checkpoint": checkpoint_path,
        "seed": seed,
        "samples": int(len(test_z)),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=5151)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data, args.device, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
