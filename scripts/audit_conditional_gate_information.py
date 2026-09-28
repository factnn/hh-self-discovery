"""Held-out low-complexity gate recovery from V, latent z, and their union."""

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


def arrays(items):
    return (
        np.concatenate([item["latents"] for item in items]),
        np.concatenate([item["voltage"] for item in items])[:, None],
        np.concatenate([item["gates"] for item in items]),
    )


def score(train_x, train_y, test_x, test_y):
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=1e-3),
    ).fit(train_x, train_y)
    values = r2_score(test_y, model.predict(test_x), multioutput="raw_values")
    return {name: float(value) for name, value in zip(("n", "m", "h"), values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--windows-per-mode", type=int, default=300)
    parser.add_argument("--seed", type=int, default=9595)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {}
    for index, checkpoint_path in enumerate(args.checkpoints):
        model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
        validation = _collect_windows(
            model, checkpoint, Path(args.data), "validation", device,
            args.windows_per_mode, args.seed + index,
        )
        test = _collect_windows(
            model, checkpoint, Path(args.data), "test", device,
            args.windows_per_mode, args.seed + 10000 + index,
        )
        z_train, v_train, g_train = arrays(validation)
        z_test, v_test, g_test = arrays(test)
        feature_sets = {
            "voltage_only": (v_train, v_test),
            "latent_only": (z_train, z_test),
            "latent_plus_voltage": (
                np.concatenate([z_train, v_train], axis=1),
                np.concatenate([z_test, v_test], axis=1),
            ),
        }
        scores = {
            name: score(x_train, g_train, x_test, g_test)
            for name, (x_train, x_test) in feature_sets.items()
        }
        scores["increment_over_voltage"] = {
            gate: scores["latent_plus_voltage"][gate] - scores["voltage_only"][gate]
            for gate in ("n", "m", "h")
        }
        scores["increment_over_latent"] = {
            gate: scores["latent_plus_voltage"][gate] - scores["latent_only"][gate]
            for gate in ("n", "m", "h")
        }
        results[Path(checkpoint_path).parent.name] = scores
    output = {"results": results, "seed": args.seed, "windows_per_mode": args.windows_per_mode}
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
