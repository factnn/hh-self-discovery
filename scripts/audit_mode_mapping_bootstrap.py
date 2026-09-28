"""Trajectory-block bootstrap for joint versus clamp-specific gate maps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint
from hh_self_discovery.simulator import gate_derivatives


def fit_map(items):
    z = np.concatenate([item["latents"] for item in items])
    gates = np.concatenate([item["gates"] for item in items])
    return make_pipeline(StandardScaler(), Ridge(alpha=0.1)).fit(z, gates)


def sample_metrics(model, mapping, items, device):
    z = np.concatenate([item["latents"] for item in items])
    gates = np.concatenate([item["gates"] for item in items])
    voltage = np.concatenate([item["voltage"] for item in items])
    groups = np.concatenate([item["trajectory_ids"] for item in items])
    mapped = mapping.predict(z)
    with torch.no_grad():
        v = torch.as_tensor(voltage, dtype=torch.float32, device=device)
        state = torch.as_tensor(z, dtype=torch.float32, device=device)
        steady, tau = model.gate_kinetics(v)
        latent_field = ((steady - state) / tau).cpu().numpy()
    epsilon = 1e-4
    pushed = (
        mapping.predict(z + epsilon * latent_field)
        - mapping.predict(z - epsilon * latent_field)
    ) / (2 * epsilon)
    true_field = np.stack([gate_derivatives(vv, gg) for vv, gg in zip(voltage, mapped)])
    cosine = np.sum(pushed * true_field, axis=1) / (
        np.linalg.norm(pushed, axis=1) * np.linalg.norm(true_field, axis=1) + 1e-12
    )
    return {
        "groups": groups,
        "gate_se": np.mean((mapped - gates) ** 2, axis=1),
        "field_se": np.mean((pushed - true_field) ** 2, axis=1),
        "field_cosine": cosine,
    }


def group_delta(joint, specific, key):
    groups = np.unique(joint["groups"])
    values = []
    for group in groups:
        mask = joint["groups"] == group
        values.append(float(np.mean(specific[key][mask] - joint[key][mask])))
    return groups, np.asarray(values)


def bootstrap(values, seed):
    rng = np.random.default_rng(seed)
    draws = np.empty(2000)
    for index in range(len(draws)):
        draws[index] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "trajectory_count": int(len(values)),
        "mean_delta_specific_minus_joint": float(values.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "trajectory_improvement_fraction": float(np.mean(values < 0)),
    }


def evaluate(checkpoint_path, data_root, device_name, seed):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    root = Path(data_root)
    validation = _collect_windows(model, checkpoint, root, "validation", device, 300, seed)
    test = _collect_windows(model, checkpoint, root, "test", device, 300, seed + 10000)
    joint_map = fit_map(validation)
    output = {}
    for mode_index, mode in enumerate(model.MODES):
        validation_mode = [item for item in validation if item["mode"] == mode]
        test_mode = [item for item in test if item["mode"] == mode]
        joint = sample_metrics(model, joint_map, test_mode, device)
        specific = sample_metrics(model, fit_map(validation_mode), test_mode, device)
        entries = {}
        for metric_index, key in enumerate(("gate_se", "field_se", "field_cosine")):
            _, delta = group_delta(joint, specific, key)
            entry = bootstrap(delta, seed + mode_index * 10 + metric_index)
            if key == "field_cosine":
                entry["trajectory_improvement_fraction"] = float(np.mean(delta > 0))
            entries[key] = entry
        output[mode] = entries
    return {"checkpoint": checkpoint_path, "bootstrap_draws": 2000, "results": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=5555)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data, args.device, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
