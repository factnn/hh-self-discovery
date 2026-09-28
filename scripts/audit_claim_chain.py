#!/usr/bin/env python3
"""Audit prediction, gate decoding, and transported fields on identical windows.

This script is deliberately post-lock.  It uses a single history encoding followed
by a free rollout, then fits low-complexity charts on validation trajectories and
evaluates all quantities on the same held-out test windows.  It also audits whether
the voltage-conditioned decoded m field is closed in (m, V), and whether explicit
voltage conditioning introduces discontinuities at voltage-clamp command jumps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint
from hh_self_discovery.metrics import regression_metrics, waveform_metrics


GATES = ("n", "m", "h")


def hh_field(voltage: np.ndarray, gates: np.ndarray) -> np.ndarray:
    """Vectorized canonical HH gate field in inverse milliseconds."""
    voltage = np.asarray(voltage, dtype=np.float64)
    gates = np.asarray(gates, dtype=np.float64)

    def vtrap(x: np.ndarray, scale: float) -> np.ndarray:
        ratio = x / scale
        regular = x / (-np.expm1(-ratio))
        series = scale * (1.0 + ratio / 2.0 + ratio * ratio / 12.0)
        return np.where(np.abs(ratio) < 1e-7, series, regular)

    an = 0.01 * vtrap(voltage + 55.0, 10.0)
    bn = 0.125 * np.exp(np.clip(-(voltage + 65.0) / 80.0, -700.0, 700.0))
    am = 0.1 * vtrap(voltage + 40.0, 10.0)
    bm = 4.0 * np.exp(np.clip(-(voltage + 65.0) / 18.0, -700.0, 700.0))
    ah = 0.07 * np.exp(np.clip(-(voltage + 65.0) / 20.0, -700.0, 700.0))
    bh = 1.0 / (1.0 + np.exp(np.clip(-(voltage + 35.0) / 10.0, -700.0, 700.0)))
    alpha = np.column_stack((an, am, ah))
    beta = np.column_stack((bn, bm, bh))
    return alpha * (1.0 - gates) - beta * gates


def fit_chart(features: np.ndarray, gates: np.ndarray, alpha: float, degree: int = 3):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        Ridge(alpha=alpha),
    ).fit(features, gates)


def gate_scores(target: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        gate: float(r2_score(target[:, index], prediction[:, index]))
        for index, gate in enumerate(GATES)
    }


def field_scores(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict:
    target = target[mask]
    prediction = prediction[mask]
    result = {"count": int(mask.sum()), "by_gate": {}}
    for index, gate in enumerate(GATES):
        scale = np.sqrt(np.mean(target[:, index] ** 2)) + 1e-12
        result["by_gate"][gate] = {
            "relative_rmse": float(
                np.sqrt(np.mean((prediction[:, index] - target[:, index]) ** 2)) / scale
            ),
            "r2": float(r2_score(target[:, index], prediction[:, index])),
            "correlation": float(np.corrcoef(target[:, index], prediction[:, index])[0, 1]),
        }
    scale = np.sqrt(np.mean(target**2)) + 1e-12
    result["overall_relative_rmse"] = float(
        np.sqrt(np.mean((prediction - target) ** 2)) / scale
    )
    return result


def flatten(items: list[dict], checkpoint: dict) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        "z": [], "gates": [], "true_v": [], "model_v": [], "true_vdot": [],
        "model_vdot": [], "mode": [], "trajectory_ids": [],
    }
    stride_steps = len(range(0, checkpoint["prediction_steps"], checkpoint["_audit_stride"]))
    dt = checkpoint["dt_ms"] * checkpoint["_audit_stride"]
    for item in items:
        z = item["latents"]
        model_v = item["model_voltage"]
        rows = len(z) // stride_steps
        model_v_rows = model_v.reshape(rows, stride_steps)
        if item["mode"] == "voltage_clamp":
            # Voltage is an imposed command in voltage clamp, not a predicted
            # response. Use the identical command derivative in both audit
            # branches; otherwise a discretization change can masquerade as a
            # true-V versus model-V comparison.
            model_vdot = item["voltage_slope"].copy()
        else:
            model_vdot = np.gradient(model_v_rows, dt, axis=1).reshape(-1)
        arrays["z"].append(z)
        arrays["gates"].append(item["gates"])
        arrays["true_v"].append(item["voltage"])
        arrays["model_v"].append(model_v)
        arrays["true_vdot"].append(item["voltage_slope"])
        arrays["model_vdot"].append(model_vdot)
        arrays["mode"].append(np.repeat(item["mode"], len(z)))
        arrays["trajectory_ids"].append(item["trajectory_ids"])
    return {key: np.concatenate(value) for key, value in arrays.items()}


def latent_field(model, z: np.ndarray, voltage: np.ndarray, device: torch.device) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, len(z), 65536):
            state = torch.as_tensor(z[start : start + 65536], dtype=torch.float32, device=device)
            v = torch.as_tensor(voltage[start : start + 65536], dtype=torch.float32, device=device)
            steady, tau = model.gate_kinetics(v)
            parts.append(((steady - state) / tau).cpu().numpy())
    return np.concatenate(parts)


def push_chart(chart, features: np.ndarray, feature_field: np.ndarray, epsilon_ms: float) -> np.ndarray:
    return (
        chart.predict(features + epsilon_ms * feature_field)
        - chart.predict(features - epsilon_ms * feature_field)
    ) / (2.0 * epsilon_ms)


def prediction_scores(items: list[dict], checkpoint: dict) -> dict:
    output = {}
    stride = checkpoint["_audit_stride"]
    for mode in ("current_clamp", "voltage_clamp"):
        subset = [item for item in items if item["mode"] == mode]
        target = np.concatenate([item["target"] for item in subset], axis=0)
        prediction = np.concatenate([item["prediction"] for item in subset], axis=0)
        activity = np.concatenate([item["activity"] for item in subset], axis=0)
        normalizer_std = checkpoint["normalizers"][mode]["response"]["std"]
        output[mode] = {
            **regression_metrics(target, prediction, activity),
            **waveform_metrics(target, prediction, mode, checkpoint["dt_ms"] * stride),
            "normalized_rmse": float(np.sqrt(np.mean((prediction - target) ** 2)) / normalizer_std),
        }
    output["mean_normalized_rmse"] = float(
        np.mean([output[mode]["normalized_rmse"] for mode in ("current_clamp", "voltage_clamp")])
    )
    return output


def closure_scores(
    train: dict[str, np.ndarray], test: dict[str, np.ndarray],
    train_mapped: np.ndarray, test_mapped: np.ndarray,
    train_pushed: np.ndarray, test_pushed: np.ndarray, seed: int,
) -> dict:
    train_mask = (train["mode"] == "voltage_clamp") & (np.abs(train["true_vdot"]) <= 1.0)
    test_mask = (test["mode"] == "voltage_clamp") & (np.abs(test["true_vdot"]) <= 1.0)
    rng = np.random.default_rng(seed)
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    train_indices = rng.choice(train_indices, size=min(60000, len(train_indices)), replace=False)
    test_indices = rng.choice(test_indices, size=min(60000, len(test_indices)), replace=False)

    train_base = np.column_stack((train_mapped[train_indices, 1], train["true_v"][train_indices]))
    test_base = np.column_stack((test_mapped[test_indices, 1], test["true_v"][test_indices]))
    train_aug = np.column_stack((train_base, train["z"][train_indices]))
    test_aug = np.column_stack((test_base, test["z"][test_indices]))
    train_target = train_pushed[train_indices, 1]
    test_target = test_pushed[test_indices, 1]

    base = ExtraTreesRegressor(
        n_estimators=160, min_samples_leaf=20, max_features=1.0,
        random_state=seed, n_jobs=4,
    ).fit(train_base, train_target)
    augmented = ExtraTreesRegressor(
        n_estimators=160, min_samples_leaf=20, max_features=1.0,
        random_state=seed + 1, n_jobs=4,
    ).fit(train_aug, train_target)
    base_r2 = float(r2_score(test_target, base.predict(test_base)))
    augmented_r2 = float(r2_score(test_target, augmented.predict(test_aug)))

    spline = SplineTransformer(n_knots=12, degree=3, include_bias=True)
    train_basis = spline.fit_transform(train["true_v"][train_indices, None])
    test_basis = spline.transform(test["true_v"][test_indices, None])
    train_affine = np.column_stack((train_basis, train_basis * train_mapped[train_indices, 1, None]))
    test_affine = np.column_stack((test_basis, test_basis * test_mapped[test_indices, 1, None]))
    affine = make_pipeline(
        StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 9))
    ).fit(train_affine, train_target)
    affine_r2 = float(r2_score(test_target, affine.predict(test_affine)))
    return {
        "definition": "voltage-clamp samples with |dV/dt| <= 1 mV/ms",
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "nonparametric_m_v_r2": base_r2,
        "nonparametric_m_v_z_r2": augmented_r2,
        "extra_latent_increment_r2": augmented_r2 - base_r2,
        "affine_in_m_voltage_spline_r2": affine_r2,
    }


def jump_scores(items: list[dict], checkpoint: dict, z_chart, zv_chart) -> dict:
    steps = len(range(0, checkpoint["prediction_steps"], checkpoint["_audit_stride"]))
    direct, actual, truth = [], [], []
    for item in items:
        if item["mode"] != "voltage_clamp":
            continue
        rows = len(item["latents"]) // steps
        z = item["latents"].reshape(rows, steps, -1)
        v = item["voltage"].reshape(rows, steps)
        gates = item["gates"].reshape(rows, steps, 3)
        for row in range(rows):
            for index in np.flatnonzero(np.abs(np.diff(v[row])) >= 1.0) + 1:
                state = z[row, index - 1]
                before = np.r_[state, v[row, index - 1]][None]
                after = np.r_[state, v[row, index]][None]
                direct.append((zv_chart.predict(after) - zv_chart.predict(before))[0])
                actual.append(
                    zv_chart.predict(np.r_[z[row, index], v[row, index]][None])[0]
                    - zv_chart.predict(before)[0]
                )
                truth.append(gates[row, index] - gates[row, index - 1])
    if not direct:
        return {"jump_count": 0}
    direct = np.asarray(direct)
    actual = np.asarray(actual)
    truth = np.asarray(truth)
    result = {"jump_count": int(len(direct)), "by_gate": {}}
    for index, gate in enumerate(GATES):
        result["by_gate"][gate] = {
            "direct_voltage_chart_jump_median_abs": float(np.median(np.abs(direct[:, index]))),
            "direct_voltage_chart_jump_p95_abs": float(np.quantile(np.abs(direct[:, index]), 0.95)),
            "actual_decoded_one_step_median_abs": float(np.median(np.abs(actual[:, index]))),
            "true_gate_one_step_median_abs": float(np.median(np.abs(truth[:, index]))),
        }
    return result


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

    z_chart = fit_chart(train["z"], train["gates"], args.ridge_alpha, args.chart_degree)
    v_chart = fit_chart(
        train["true_v"][:, None], train["gates"], args.ridge_alpha, args.chart_degree
    )
    zv_chart = fit_chart(
        np.column_stack((train["z"], train["true_v"])), train["gates"], args.ridge_alpha,
        args.chart_degree,
    )
    train_features = np.column_stack((train["z"], train["true_v"]))
    test_features = np.column_stack((test["z"], test["true_v"]))
    train_mapped = zv_chart.predict(train_features)
    test_mapped = zv_chart.predict(test_features)
    z_only_mapped = z_chart.predict(test["z"])
    voltage_only_mapped = v_chart.predict(test["true_v"][:, None])

    train_latent_field = latent_field(model, train["z"], train["true_v"], device)
    test_latent_field = latent_field(model, test["z"], test["true_v"], device)
    train_pushed = push_chart(
        zv_chart, train_features,
        np.column_stack((train_latent_field, train["true_vdot"])), args.jvp_epsilon_ms,
    )
    test_pushed = push_chart(
        zv_chart, test_features,
        np.column_stack((test_latent_field, test["true_vdot"])), args.jvp_epsilon_ms,
    )
    true_field = hh_field(test["true_v"], test["gates"])

    model_features = np.column_stack((test["z"], test["model_v"]))
    model_latent_field = latent_field(model, test["z"], test["model_v"], device)
    model_mapped = zv_chart.predict(model_features)
    model_pushed = push_chart(
        zv_chart, model_features,
        np.column_stack((model_latent_field, test["model_vdot"])), args.jvp_epsilon_ms,
    )

    masks = {
        "all": np.ones(len(test["z"]), dtype=bool),
        "current_clamp": test["mode"] == "current_clamp",
        "voltage_clamp": test["mode"] == "voltage_clamp",
        "voltage_clamp_smooth": (test["mode"] == "voltage_clamp") & (np.abs(test["true_vdot"]) <= 1.0),
    }
    return {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "seed": args.seed,
        "windows_per_mode": args.windows_per_mode,
        "stride": args.stride,
        "ridge_alpha": args.ridge_alpha,
        "jvp_epsilon_ms": args.jvp_epsilon_ms,
        "latent_source": "one observed history encoding followed by free rollout",
        "local_audit": "free-roll z paired with observed V and observed dV/dt",
        "prediction_same_windows": prediction_scores(test_items, checkpoint),
        "state_same_windows": {
            "z_only_gate_r2": gate_scores(test["gates"], z_only_mapped),
            "voltage_only_gate_r2": gate_scores(test["gates"], voltage_only_mapped),
            "z_observed_or_command_voltage_gate_r2": gate_scores(
                test["gates"], test_mapped
            ),
            "z_rollout_or_command_voltage_gate_r2": gate_scores(
                test["gates"], model_mapped
            ),
            "by_mode": {
                mode: {
                    "z_only_gate_r2": gate_scores(
                        test["gates"][test["mode"] == mode],
                        z_only_mapped[test["mode"] == mode],
                    ),
                    "voltage_only_gate_r2": gate_scores(
                        test["gates"][test["mode"] == mode],
                        voltage_only_mapped[test["mode"] == mode],
                    ),
                    "z_observed_or_command_voltage_gate_r2": gate_scores(
                        test["gates"][test["mode"] == mode],
                        test_mapped[test["mode"] == mode],
                    ),
                    "z_rollout_or_command_voltage_gate_r2": gate_scores(
                        test["gates"][test["mode"] == mode],
                        model_mapped[test["mode"] == mode],
                    ),
                }
                for mode in ("current_clamp", "voltage_clamp")
            },
            "paired_support": {
                "voltage_clamp_smooth": {
                    "definition": (
                        "voltage-clamp samples with |dV_cmd/dt| <= 1 mV/ms"
                    ),
                    "count": int(masks["voltage_clamp_smooth"].sum()),
                    "z_only_gate_r2": gate_scores(
                        test["gates"][masks["voltage_clamp_smooth"]],
                        z_only_mapped[masks["voltage_clamp_smooth"]],
                    ),
                    "voltage_only_gate_r2": gate_scores(
                        test["gates"][masks["voltage_clamp_smooth"]],
                        voltage_only_mapped[masks["voltage_clamp_smooth"]],
                    ),
                    "z_command_voltage_gate_r2": gate_scores(
                        test["gates"][masks["voltage_clamp_smooth"]],
                        test_mapped[masks["voltage_clamp_smooth"]],
                    ),
                }
            },
            "voltage_semantics": {
                "current_clamp": "observed response V versus freely predicted response V_hat",
                "voltage_clamp": "both branches use the identical imposed command V_cmd",
            },
        },
        "transported_field_same_windows": {
            "observed_or_command_voltage_local": {
                name: field_scores(true_field, test_pushed, mask) for name, mask in masks.items()
            },
            "rollout_or_command_voltage": {
                name: field_scores(true_field, model_pushed, mask) for name, mask in masks.items()
            },
            "voltage_semantics": {
                "current_clamp": (
                    "first branch uses observed response V; second uses freely "
                    "predicted response V_hat"
                ),
                "voltage_clamp": "both branches use the same V_cmd and dV_cmd/dt",
            },
        },
        "m_closure": closure_scores(
            train, test, train_mapped, test_mapped, train_pushed, test_pushed, args.seed
        ),
        "voltage_jump_continuity": jump_scores(test_items, checkpoint, z_chart, zv_chart),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19600)
    parser.add_argument("--windows-per-mode", type=int, default=200)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--jvp-epsilon-ms", type=float, default=0.01)
    parser.add_argument("--chart-degree", type=int, default=3)
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
