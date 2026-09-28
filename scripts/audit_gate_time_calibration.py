"""Test whether a low-complexity time rescaling repairs mapped gate dynamics."""

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


GATES = ("n", "m", "h")


def metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    result = {}
    for index, gate in enumerate(GATES):
        scale = np.sqrt(np.mean(target[:, index] ** 2)) + 1e-12
        result[gate] = {
            "relative_rmse": float(
                np.sqrt(np.mean((predicted[:, index] - target[:, index]) ** 2)) / scale
            ),
            "pearson": float(np.corrcoef(predicted[:, index], target[:, index])[0, 1]),
            "sign_agreement": float(
                np.mean(np.sign(predicted[:, index]) == np.sign(target[:, index]))
            ),
        }
    scale = np.sqrt(np.mean(target**2)) + 1e-12
    cosine = np.sum(predicted * target, axis=1) / (
        np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1) + 1e-12
    )
    return {
        "overall_relative_rmse": float(np.sqrt(np.mean((predicted - target) ** 2)) / scale),
        "overall_mean_cosine": float(np.mean(cosine)),
        "by_gate": result,
    }


def fields(model, readout, items, device, seed):
    z = np.concatenate([item["latents"] for item in items])
    voltage = np.concatenate([item["voltage"] for item in items])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(z), size=min(40000, len(z)), replace=False)
    z, voltage = z[chosen], voltage[chosen]
    with torch.no_grad():
        voltage_tensor = torch.as_tensor(voltage, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(z, dtype=torch.float32, device=device)
        steady, tau = model.gate_kinetics(voltage_tensor)
        latent_field = ((steady - state_tensor) / tau).cpu().numpy()
    gates = readout.predict(z)
    true_field = np.stack([gate_derivatives(v, g) for v, g in zip(voltage, gates)])
    epsilon = 1e-4
    pushed = (
        readout.predict(z + epsilon * latent_field)
        - readout.predict(z - epsilon * latent_field)
    ) / (2.0 * epsilon)
    valid = np.all((gates >= 0.0) & (gates <= 1.0), axis=1)
    return voltage[valid], pushed[valid], true_field[valid]


def evaluate(checkpoint_path: str, data_root: str, device_name: str, seed: int) -> dict:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    root = Path(data_root)
    validation = _collect_windows(model, checkpoint, root, "validation", device, 300, seed)
    test = _collect_windows(model, checkpoint, root, "test", device, 300, seed + 10000)
    validation_z = np.concatenate([item["latents"] for item in validation])
    validation_g = np.concatenate([item["gates"] for item in validation])
    readout = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=1e-3),
    ).fit(validation_z, validation_g)
    val_v, val_pushed, val_true = fields(model, readout, validation, device, seed + 1)
    test_v, test_pushed, test_true = fields(model, readout, test, device, seed + 2)
    constant_pred = np.empty_like(test_pushed)
    voltage_pred = np.empty_like(test_pushed)
    constant_scale, voltage_coefficients = {}, {}
    val_scaled = (val_v + 50.0) / 100.0
    test_scaled = (test_v + 50.0) / 100.0
    basis_val = np.column_stack([np.ones(len(val_v)), val_scaled, val_scaled**2, val_scaled**3])
    basis_test = np.column_stack(
        [np.ones(len(test_v)), test_scaled, test_scaled**2, test_scaled**3]
    )
    shared_constant = float(
        np.sum(val_pushed * val_true) / (np.sum(val_pushed**2) + 1e-12)
    )
    shared_constant_pred = shared_constant * test_pushed
    shared_design = np.concatenate(
        [basis_val * val_pushed[:, index, None] for index in range(len(GATES))]
    )
    shared_target = np.concatenate(
        [val_true[:, index] for index in range(len(GATES))]
    )
    shared_estimator = Ridge(alpha=1e-6, fit_intercept=False).fit(
        shared_design, shared_target
    )
    shared_voltage_scale = basis_test @ shared_estimator.coef_
    shared_voltage_pred = test_pushed * shared_voltage_scale[:, None]
    for index, gate in enumerate(GATES):
        source = val_pushed[:, index]
        target = val_true[:, index]
        scale = float(np.dot(source, target) / (np.dot(source, source) + 1e-12))
        constant_scale[gate] = scale
        constant_pred[:, index] = scale * test_pushed[:, index]
        design = basis_val * source[:, None]
        estimator = Ridge(alpha=1e-6, fit_intercept=False).fit(design, target)
        voltage_coefficients[gate] = estimator.coef_.tolist()
        voltage_pred[:, index] = (
            basis_test * test_pushed[:, index, None]
        ) @ estimator.coef_
    return {
        "checkpoint": checkpoint_path,
        "seed": seed,
        "validation_samples": int(len(val_v)),
        "test_samples": int(len(test_v)),
        "constant_scale": constant_scale,
        "voltage_scale_coefficients": voltage_coefficients,
        "shared_constant_scale": shared_constant,
        "shared_voltage_scale_coefficients": shared_estimator.coef_.tolist(),
        "test": {
            "uncalibrated": metrics(test_pushed, test_true),
            "shared_constant_time_scale": metrics(shared_constant_pred, test_true),
            "shared_cubic_voltage_time_scale": metrics(shared_voltage_pred, test_true),
            "constant_time_scale": metrics(constant_pred, test_true),
            "cubic_voltage_time_scale": metrics(voltage_pred, test_true),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=4949)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data, args.device, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
