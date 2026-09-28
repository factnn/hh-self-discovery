#!/usr/bin/env python3
"""Capacity control for the closure regression, with a known-coordinate control.

The post-lock closure test asks whether the decoded fast gate, together with
voltage, determines its own derivative.  It predicts the transported field from
``(m_hat, V)`` and then from ``(m_hat, V, z)``, and reports the increment.  This
script answers two questions about that test:

* is the increment an artefact of the regressor's capacity?  The same
  comparison is repeated across tree counts, leaf sizes, and a different model
  class (histogram gradient boosting);
* does the increment also appear when the decode is known to be correct?  The
  known-coordinate control replaces the learned latents with the true HH gates
  behind a known invertible transform, where ``m_hat`` recovers ``m`` almost
  exactly and the HH ``m`` derivative depends on ``m`` and ``V`` alone.  A
  well-behaved closure test must therefore report no increment there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_claim_chain import fit_chart, flatten, hh_field, latent_field, push_chart  # noqa: E402
from audit_known_coordinate_control import transformed  # noqa: E402

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint  # noqa: E402

DEFAULT_TRAIN_CAP = 20_000
DEFAULT_TEST_CAP = 20_000


def sampled(mask: np.ndarray, seed: int, cap: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=min(cap, len(indices)), replace=False)


def fit_pair(
    train_base: np.ndarray, train_aug: np.ndarray, target_train: np.ndarray,
    test_base: np.ndarray, test_aug: np.ndarray, target_test: np.ndarray,
    config: dict, seed: int,
) -> dict:
    if config["kind"] == "extra_trees":
        kwargs = dict(
            n_estimators=config["n_estimators"], min_samples_leaf=config["min_samples_leaf"],
            max_features=1.0, random_state=seed, n_jobs=8,
        )
        base = ExtraTreesRegressor(**kwargs).fit(train_base, target_train)
        augmented = ExtraTreesRegressor(**{**kwargs, "random_state": seed + 1}).fit(
            train_aug, target_train
        )
    else:
        kwargs = dict(max_iter=config["max_iter"], min_samples_leaf=config["min_samples_leaf"],
                      random_state=seed)
        base = HistGradientBoostingRegressor(**kwargs).fit(train_base, target_train)
        augmented = HistGradientBoostingRegressor(**{**kwargs, "random_state": seed + 1}).fit(
            train_aug, target_train
        )
    base_r2 = float(r2_score(target_test, base.predict(test_base)))
    augmented_r2 = float(r2_score(target_test, augmented.predict(test_aug)))
    return {
        "config": config,
        "m_v_r2": base_r2,
        "m_v_z_r2": augmented_r2,
        "increment_r2": augmented_r2 - base_r2,
    }


def closure_datasets(arrays: dict, mapped: np.ndarray, pushed: np.ndarray, index: np.ndarray) -> dict:
    base = np.column_stack((mapped[index, 1], arrays["true_v"][index]))
    augmented = np.column_stack((base, arrays["z"][index]))
    return {"base": base, "augmented": augmented}


CONFIGS = [
    {"kind": "extra_trees", "n_estimators": 160, "min_samples_leaf": 20},
    {"kind": "extra_trees", "n_estimators": 160, "min_samples_leaf": 5},
    {"kind": "extra_trees", "n_estimators": 160, "min_samples_leaf": 50},
    {"kind": "extra_trees", "n_estimators": 640, "min_samples_leaf": 20},
    {"kind": "hist_gradient_boosting", "max_iter": 300, "min_samples_leaf": 20},
]


def evaluate(args) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(args.checkpoint, device)
    checkpoint = dict(checkpoint)
    checkpoint["_audit_stride"] = args.stride
    root = Path(args.data)
    validation_items = _collect_windows(
        model, checkpoint, root, "validation", device, args.windows_per_mode,
        args.seed, stride=args.stride,
    )
    test_items = _collect_windows(
        model, checkpoint, root, "test", device, args.windows_per_mode,
        args.seed + 10000, stride=args.stride,
    )
    train = flatten(validation_items, checkpoint)
    test = flatten(test_items, checkpoint)

    train_features = np.column_stack((train["z"], train["true_v"]))
    test_features = np.column_stack((test["z"], test["true_v"]))
    chart = fit_chart(train_features, train["gates"], args.ridge_alpha, args.chart_degree)
    train_mapped = chart.predict(train_features)
    test_mapped = chart.predict(test_features)
    train_pushed = push_chart(
        chart, train_features,
        np.column_stack((latent_field(model, train["z"], train["true_v"], device), train["true_vdot"])),
        args.jvp_epsilon_ms,
    )
    test_pushed = push_chart(
        chart, test_features,
        np.column_stack((latent_field(model, test["z"], test["true_v"], device), test["true_vdot"])),
        args.jvp_epsilon_ms,
    )

    smooth_train = (train["mode"] == "voltage_clamp") & (np.abs(train["true_vdot"]) <= 1.0)
    smooth_test = (test["mode"] == "voltage_clamp") & (np.abs(test["true_vdot"]) <= 1.0)
    train_index = sampled(smooth_train, args.seed, args.train_cap)
    test_index = sampled(smooth_test, args.seed + 1, args.test_cap)

    datasets = {}
    # (1) Learned model: target is the transported m field of the fitted model-chart pair.
    learned = closure_datasets(train, train_mapped, train_pushed, train_index)
    learned_test = closure_datasets(test, test_mapped, test_pushed, test_index)
    datasets["learned"] = (learned, learned_test, train_pushed[train_index, 1], test_pushed[test_index, 1])

    # (2) Known-coordinate controls: the latents are transformed true gates and
    # the target is the exact HH m field.
    for kind in ("affine", "nonlinear_logit_mix"):
        train_field = hh_field(train["true_v"], train["gates"])
        test_field = hh_field(test["true_v"], test["gates"])
        train_z, _ = transformed(train["gates"], train_field, kind)
        test_z, _ = transformed(test["gates"], test_field, kind)
        control_train = {**train, "z": train_z}
        control_test = {**test, "z": test_z}
        control_chart = fit_chart(
            np.column_stack((train_z, train["true_v"])), train["gates"],
            args.ridge_alpha, args.chart_degree,
        )
        control_train_mapped = control_chart.predict(np.column_stack((train_z, train["true_v"])))
        control_test_mapped = control_chart.predict(np.column_stack((test_z, test["true_v"])))
        datasets[f"control_{kind}"] = (
            closure_datasets(control_train, control_train_mapped, train_field, train_index),
            closure_datasets(control_test, control_test_mapped, test_field, test_index),
            train_field[train_index, 1],
            test_field[test_index, 1],
        )

    report = {}
    for name, (train_set, test_set, target_train, target_test) in datasets.items():
        report[name] = {
            "train_samples": int(len(train_set["base"])),
            "test_samples": int(len(test_set["base"])),
            "fits": [
                fit_pair(
                    train_set["base"], train_set["augmented"], target_train,
                    test_set["base"], test_set["augmented"], target_test,
                    config, args.seed,
                )
                for config in CONFIGS
            ],
        }
    return {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "seed": args.seed,
        "stride": args.stride,
        "ridge_alpha": args.ridge_alpha,
        "chart_degree": args.chart_degree,
        "target": "transported m field on smooth voltage-clamp samples",
        "closure_features": "(m_hat, V) against (m_hat, V, z)",
        "configs": CONFIGS,
        "datasets": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19800)
    parser.add_argument("--windows-per-mode", type=int, default=200)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--chart-degree", type=int, default=3)
    parser.add_argument("--jvp-epsilon-ms", type=float, default=0.01)
    parser.add_argument("--train-cap", type=int, default=DEFAULT_TRAIN_CAP)
    parser.add_argument("--test-cap", type=int, default=DEFAULT_TEST_CAP)
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
