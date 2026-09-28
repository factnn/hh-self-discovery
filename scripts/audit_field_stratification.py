#!/usr/bin/env python3
"""Where the fast-gate field error lives, and how well conditioned the chart is.

Two post-lock diagnostics on the same smooth voltage-clamp support:

* the transported-field error stratified by the local HH time constant
  ``tau_m(V)`` and by the magnitude of the true ``m`` derivative, so the fast
  gate is identified by a continuous physical quantity rather than by the
  ``n/m/h`` label;
* the conditioning of the fitted chart's latent Jacobian ``d r / d z``, and its
  rank correlation with the pointwise field error, which tests whether the
  residual disagreement is an artefact of a pathological chart geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_claim_chain import fit_chart, flatten, hh_field, latent_field, push_chart  # noqa: E402

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint  # noqa: E402

GATES = ("n", "m", "h")


def vtrap(x: np.ndarray, scale: float) -> np.ndarray:
    ratio = x / scale
    regular = x / (-np.expm1(-ratio))
    series = scale * (1.0 + ratio / 2.0 + ratio * ratio / 12.0)
    return np.where(np.abs(ratio) < 1e-7, series, regular)


def tau_m(voltage: np.ndarray) -> np.ndarray:
    alpha = 0.1 * vtrap(voltage + 40.0, 10.0)
    beta = 4.0 * np.exp(np.clip(-(voltage + 65.0) / 18.0, -700.0, 700.0))
    return 1.0 / (alpha + beta)


def quartile_report(values: np.ndarray, error: np.ndarray, pushed: np.ndarray, truth: np.ndarray) -> list[dict]:
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    for low, high, index in zip(edges[:-1], edges[1:], range(4)):
        selected = (values >= low) & (values <= high) if index == 3 else (values >= low) & (values < high)
        if selected.sum() < 50:
            continue
        guess, target = pushed[selected], truth[selected]
        slope = float(np.cov(guess, target, ddof=1)[0, 1] / (np.var(target, ddof=1) + 1e-24))
        rows.append({
            "range": [float(low), float(high)],
            "count": int(selected.sum()),
            "mean_abs_error": float(error[selected].mean()),
            "relative_rmse": float(
                np.sqrt(np.mean((guess - target) ** 2)) / (np.sqrt(np.mean(target**2)) + 1e-12)
            ),
            "correlation": float(np.corrcoef(guess, target)[0, 1]),
            "slope": slope,
        })
    return rows


def jacobian_diagnostics(chart, features: np.ndarray, latent_width: int, step: float = 1e-3) -> dict:
    """Singular values of the chart's latent Jacobian, in raw feature space."""
    latent = features[:, :latent_width]
    columns = []
    for index in range(latent_width):
        delta = np.zeros_like(latent)
        delta[:, index] = step
        plus = chart.predict(np.column_stack((latent + delta, features[:, latent_width:])))
        minus = chart.predict(np.column_stack((latent - delta, features[:, latent_width:])))
        columns.append((plus - minus) / (2.0 * step))
    jacobian = np.stack(columns, axis=2)          # samples x gates x latent
    singular = np.linalg.svd(jacobian, compute_uv=False)
    largest, smallest = singular[:, 0], singular[:, -1]
    return {
        "largest_singular_median": float(np.median(largest)),
        "smallest_singular_median": float(np.median(smallest)),
        "smallest_singular_p05": float(np.quantile(smallest, 0.05)),
        "condition_number_median": float(np.median(largest / np.maximum(smallest, 1e-12))),
        "condition_number_p95": float(np.quantile(largest / np.maximum(smallest, 1e-12), 0.95)),
        "per_sample": {"largest": largest, "smallest": smallest},
    }


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
    test_mapped = chart.predict(test_features)
    test_pushed = push_chart(
        chart, test_features,
        np.column_stack((latent_field(model, test["z"], test["true_v"], device), test["true_vdot"])),
        args.jvp_epsilon_ms,
    )
    true_field = hh_field(test["true_v"], test["gates"])
    smooth = (test["mode"] == "voltage_clamp") & (np.abs(test["true_vdot"]) <= 1.0)

    index = np.flatnonzero(smooth)
    tau = tau_m(test["true_v"][index])
    derivative = np.abs(true_field[index, 1])
    error = np.abs(test_pushed[index, 1] - true_field[index, 1])
    pushed_m, true_m = test_pushed[index, 1], true_field[index, 1]

    jacobian = jacobian_diagnostics(chart, test_features[index], test["z"].shape[1])
    per_sample = jacobian.pop("per_sample")
    conditioning = {
        **jacobian,
        "spearman_condition_vs_m_error": float(
            spearmanr(per_sample["largest"] / np.maximum(per_sample["smallest"], 1e-12), error).statistic
        ),
        "spearman_smallest_singular_vs_m_error": float(
            spearmanr(per_sample["smallest"], error).statistic
        ),
    }
    return {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "seed": args.seed,
        "stride": args.stride,
        "ridge_alpha": args.ridge_alpha,
        "samples": int(len(index)),
        "m_field_r2": float(r2_score(true_m, pushed_m)),
        "by_tau_m_quartile": quartile_report(tau, error, pushed_m, true_m),
        "by_abs_true_derivative_quartile": quartile_report(derivative, error, pushed_m, true_m),
        "chart_jacobian": conditioning,
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
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
