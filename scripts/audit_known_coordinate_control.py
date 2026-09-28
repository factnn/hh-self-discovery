#!/usr/bin/env python3
"""End-to-end positive control for the post-lock chart and JVP audit.

Canonical HH gates are hidden behind known affine and smooth nonlinear invertible
coordinate transforms.  The same cubic voltage-conditioned chart and centered
JVP used by the claim audit must recover both gate values and the transported HH
field.  Failure here measures audit-induced error rather than learner error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_claim_chain import field_scores, fit_chart, flatten, gate_scores, hh_field, push_chart
from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint


AFFINE_A = np.asarray([
    [0.90, 0.18, -0.12],
    [-0.14, 1.05, 0.16],
    [0.10, -0.20, 0.94],
], dtype=np.float64)
AFFINE_B = np.asarray([0.08, -0.05, 0.04], dtype=np.float64)
NONLINEAR_A = np.asarray([
    [1.00, 0.16, -0.10],
    [-0.12, 0.92, 0.14],
    [0.08, -0.15, 1.06],
], dtype=np.float64)
NONLINEAR_B = np.asarray([0.10, -0.08, 0.06], dtype=np.float64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def transformed(gates: np.ndarray, gate_field: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "affine":
        return gates @ AFFINE_A.T + AFFINE_B, gate_field @ AFFINE_A.T
    clipped = np.clip(gates, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped))
    latent = sigmoid(logits @ NONLINEAR_A.T + NONLINEAR_B)
    logit_field = gate_field / (clipped * (1.0 - clipped))
    preactivation_field = logit_field @ NONLINEAR_A.T
    latent_field = latent * (1.0 - latent) * preactivation_field
    return latent, latent_field


def exact_inverse(latent: np.ndarray, kind: str) -> np.ndarray:
    if kind == "affine":
        return (latent - AFFINE_B) @ np.linalg.inv(AFFINE_A).T
    clipped = np.clip(latent, 1e-7, 1.0 - 1e-7)
    latent_logits = np.log(clipped / (1.0 - clipped))
    gate_logits = (latent_logits - NONLINEAR_B) @ np.linalg.inv(NONLINEAR_A).T
    return sigmoid(gate_logits)


def evaluate_transform(
    kind: str, train: dict[str, np.ndarray], test: dict[str, np.ndarray],
    ridge_alpha: float, epsilon_ms: float,
) -> dict:
    train_true_field = hh_field(train["true_v"], train["gates"])
    test_true_field = hh_field(test["true_v"], test["gates"])
    train_z, train_zdot = transformed(train["gates"], train_true_field, kind)
    test_z, test_zdot = transformed(test["gates"], test_true_field, kind)
    train_features = np.column_stack((train_z, train["true_v"]))
    test_features = np.column_stack((test_z, test["true_v"]))
    chart = fit_chart(train_features, train["gates"], ridge_alpha)
    mapped = chart.predict(test_features)
    pushed = push_chart(
        chart, test_features, np.column_stack((test_zdot, test["true_vdot"])), epsilon_ms
    )
    exact = exact_inverse(test_z, kind)
    masks = {
        "all": np.ones(len(test_z), dtype=bool),
        "current_clamp": test["mode"] == "current_clamp",
        "voltage_clamp": test["mode"] == "voltage_clamp",
        "voltage_clamp_smooth": (
            (test["mode"] == "voltage_clamp") & (np.abs(test["true_vdot"]) <= 1.0)
        ),
    }
    return {
        "transform": kind,
        "matrix_condition_number": float(np.linalg.cond(AFFINE_A if kind == "affine" else NONLINEAR_A)),
        "exact_inverse_gate_r2": gate_scores(test["gates"], exact),
        "fitted_chart_gate_r2": gate_scores(test["gates"], mapped),
        "fitted_chart_field": {
            name: field_scores(test_true_field, pushed, mask) for name, mask in masks.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Used only to reproduce the standard window sampler.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19700)
    parser.add_argument("--windows-per-mode", type=int, default=300)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--jvp-epsilon-ms", type=float, default=0.01)
    args = parser.parse_args()

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
    result = {
        "checkpoint_used_for_window_sampling": args.checkpoint,
        "data": args.data,
        "seed": args.seed,
        "ridge_alpha": args.ridge_alpha,
        "jvp_epsilon_ms": args.jvp_epsilon_ms,
        "controls": {
            kind: evaluate_transform(kind, train, test, args.ridge_alpha, args.jvp_epsilon_ms)
            for kind in ("affine", "nonlinear_logit_mix")
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
