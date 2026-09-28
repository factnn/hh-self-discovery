#!/usr/bin/env python3
"""One-step gate-flow audit for discrete predictors and for the structured model.

The transported-field audit in ``scripts/audit_claim_chain.py`` needs a latent
*velocity*, which a discrete sequence model such as S4D does not have.  This
script supplies one and keeps every other element of the audit fixed, so the
result is comparable across architectures:

* the latent velocity is estimated from the model's own one-step map,
  ``dz/dt = (z_{t+1} - z_t) / dt``, evaluated on the model's free rollout;
* the same cubic ridge chart, fitted on the same in-distribution validation
  windows with the same coefficient, transports that velocity through the
  chain rule ``J_z r . dz/dt + d_V r . dV/dt`` by central differences;
* the target is the canonical HH gate field on the same samples.

For a continuous model the script additionally reports the published analytic
velocity (``dz/dt`` from the model's own kinetics), which checks that the
finite-difference velocity reproduces the published transported-field score.

The script also runs a derivative-free comparison on the same samples. From the
decoded state ``r(z_t, V)`` it compares the model's own next decoded state
``r(z_{t+1}, V)`` with the exact HH one-step relaxation
``x_inf(V) + (r(z_t, V) - x_inf(V)) exp(-dt / tau(V))``, which is exact under
voltage clamp because the command is constant over a step.  This answers whether
the model's next step agrees with the HH next step without any Jacobian or
instantaneous-derivative approximation.

Two secondary quantities are reported because they answer a different question:

* ``increment_*``: the decoded one-step gate change ``r(z_{t+1}, V_{t+1}) -
  r(z_t, V_t)`` against the true HH change over the same interval.  This
  includes the chart's nonlinearity over the step, so it is not the transported
  field; ``increment_generator_at_true`` shows how much of the true change the
  linearised field itself misses at that interval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_claim_chain import (  # noqa: E402
    GATES,
    fit_chart,
    gate_scores,
    hh_field,
    latent_field,
    push_chart,
)

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint  # noqa: E402


def hh_kinetics(voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonical HH steady state and time constant for n, m, h at a given voltage."""

    def vtrap(x: np.ndarray, scale: float) -> np.ndarray:
        ratio = x / scale
        regular = x / (-np.expm1(-ratio))
        series = scale * (1.0 + ratio / 2.0 + ratio * ratio / 12.0)
        return np.where(np.abs(ratio) < 1e-7, series, regular)

    voltage = np.asarray(voltage, dtype=np.float64)
    alpha = np.column_stack((
        0.01 * vtrap(voltage + 55.0, 10.0),
        0.1 * vtrap(voltage + 40.0, 10.0),
        0.07 * np.exp(np.clip(-(voltage + 65.0) / 20.0, -700.0, 700.0)),
    ))
    beta = np.column_stack((
        0.125 * np.exp(np.clip(-(voltage + 65.0) / 80.0, -700.0, 700.0)),
        4.0 * np.exp(np.clip(-(voltage + 65.0) / 18.0, -700.0, 700.0)),
        1.0 / (1.0 + np.exp(np.clip(-(voltage + 35.0) / 10.0, -700.0, 700.0))),
    ))
    return alpha / (alpha + beta), 1.0 / (alpha + beta)


def steps_for(checkpoint: dict, stride: int) -> int:
    return len(range(0, checkpoint["prediction_steps"], stride))


def pair_arrays(items: list[dict], checkpoint: dict, stride: int) -> dict[str, np.ndarray]:
    """Per-pair features, velocities, increments, and evaluation-only gates.

    Consecutive samples of the flattened arrays can belong to different windows,
    so windows are restored from the stored stride length before pairing.
    """
    steps = steps_for(checkpoint, stride)
    dt = checkpoint["dt_ms"] * stride
    collected: dict[str, list[np.ndarray]] = {
        key: [] for key in
        ("features", "features_next", "latent_velocity", "voltage_velocity", "voltage",
         "gates", "true_increment", "slope", "mode")
    }
    for item in items:
        width = item["latents"].shape[-1]
        rows = len(item["latents"]) // steps
        z = item["latents"].reshape(rows, steps, width)
        voltage = item["voltage"].reshape(rows, steps)
        gates = item["gates"].reshape(rows, steps, len(GATES))
        voltage_slope = item["voltage_slope"].reshape(rows, steps)
        features = np.concatenate((z, voltage[..., None]), axis=-1)

        previous, following = slice(0, steps - 1), slice(1, steps)
        collected["features"].append(features[:, previous].reshape(-1, width + 1))
        collected["features_next"].append(features[:, following].reshape(-1, width + 1))
        collected["latent_velocity"].append(
            ((z[:, following] - z[:, previous]) / dt).reshape(-1, width)
        )
        collected["voltage_velocity"].append(
            ((voltage[:, following] - voltage[:, previous]) / dt).reshape(-1)
        )
        collected["voltage"].append(voltage[:, previous].reshape(-1))
        collected["gates"].append(gates[:, previous].reshape(-1, len(GATES)))
        collected["true_increment"].append(
            (gates[:, following] - gates[:, previous]).reshape(-1, len(GATES))
        )
        collected["slope"].append(voltage_slope[:, previous].reshape(-1))
        collected["mode"].append(np.repeat(item["mode"], rows * (steps - 1)))
    return {key: np.concatenate(value) for key, value in collected.items()}


def score(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray, what: str) -> dict:
    result = {"count": int(mask.sum()), what: {}}
    for index, gate in enumerate(GATES):
        truth = target[mask, index]
        estimate = prediction[mask, index]
        scale = float(np.sqrt(np.mean(truth**2))) + 1e-12
        covariance = float(np.cov(estimate, truth, ddof=1)[0, 1])
        variance = float(np.var(truth, ddof=1)) + 1e-24
        correlation = float(np.corrcoef(truth, estimate)[0, 1])
        result[what][gate] = {
            "r2": float(r2_score(truth, estimate)),
            "r2_after_rescaling": correlation**2,
            "slope": covariance / variance,
            "relative_rmse": float(np.sqrt(np.mean((estimate - truth) ** 2)) / scale),
            "correlation": correlation,
            "target_rms": scale,
        }
    return result


def evaluate(args) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(args.checkpoint, device)
    checkpoint = dict(checkpoint)
    root = Path(args.data)
    evaluation_root = Path(args.evaluation_data) if args.evaluation_data else root
    continuous = hasattr(model, "gate_kinetics")

    blocks = {}
    for stride in (args.native_stride, args.published_stride):
        checkpoint["_audit_stride"] = stride
        validation_items = _collect_windows(
            model, checkpoint, root, "validation", device, args.windows_per_mode,
            args.seed, stride=stride,
        )
        test_items = _collect_windows(
            model, checkpoint, evaluation_root, "test", device, args.windows_per_mode,
            args.seed + 10000, stride=stride, strict_sampling=evaluation_root == root,
        )
        train_features, train_gates = [], []
        test_mapped_chunks = []
        steps = steps_for(checkpoint, stride)
        width = validation_items[0]["latents"].shape[-1]
        for item in validation_items:
            rows = len(item["latents"]) // steps
            z = item["latents"].reshape(rows, steps, width)
            voltage = item["voltage"].reshape(rows, steps)
            train_features.append(np.concatenate((z, voltage[..., None]), axis=-1).reshape(-1, width + 1))
            train_gates.append(item["gates"])
        chart = fit_chart(
            np.concatenate(train_features), np.concatenate(train_gates),
            args.ridge_alpha, args.chart_degree,
        )
        for item in test_items:
            rows = len(item["latents"]) // steps
            z = item["latents"].reshape(rows, steps, width)
            voltage = item["voltage"].reshape(rows, steps)
            test_mapped_chunks.append(
                chart.predict(
                    np.concatenate((z, voltage[..., None]), axis=-1).reshape(-1, width + 1)
                ).reshape(rows, steps, len(GATES))
            )

        records = pair_arrays(test_items, checkpoint, stride)
        # Decoded one-step change of the model's own gate trajectory.
        decoded = np.concatenate([
            (mapped[:, 1:] - mapped[:, :-1]).reshape(-1, len(GATES))
            for mapped in test_mapped_chunks
        ])
        dt = checkpoint["dt_ms"] * stride
        field_discrete = push_chart(
            chart, records["features"],
            np.column_stack((records["latent_velocity"], records["voltage_velocity"])),
            args.jvp_epsilon_ms,
        )
        field_analytic = (
            push_chart(
                chart, records["features"],
                np.column_stack((
                    latent_field(model, records["features"][:, :width], records["voltage"], device),
                    records["slope"],
                )),
                args.jvp_epsilon_ms,
            )
            if continuous else None
        )
        true_field = hh_field(records["voltage"], records["gates"])
        smooth = (
            (records["mode"] == "voltage_clamp")
            # The interval itself must be smooth: a command jump between the two
            # samples of a pair would otherwise put an enormous voltage velocity
            # into the chain rule.  ``slope`` is the backward difference at the
            # opening sample, matching the published support definition.
            & (np.abs(records["slope"]) <= args.smooth_slope_limit)
            & (np.abs(records["voltage_velocity"]) <= args.smooth_slope_limit)
        )
        block = {
            "dt_ms": dt,
            "support": "voltage clamp with |dV_cmd/dt| <= %.1f mV/ms" % args.smooth_slope_limit,
            "state_r2_on_smooth_support": gate_scores(
                records["gates"][smooth],
                chart.predict(records["features"])[smooth],
            ),
            "field_from_discrete_velocity": score(true_field, field_discrete, smooth, "by_gate"),
            "increment_decoded": score(records["true_increment"], decoded, smooth, "by_gate"),
            "increment_generator_at_decoded": score(
                records["true_increment"],
                hh_field(records["voltage"], chart.predict(records["features"])) * dt,
                smooth, "by_gate",
            ),
            "increment_generator_at_true": score(
                records["true_increment"], true_field * dt, smooth, "by_gate",
            ),
            "increment_zero": score(
                records["true_increment"], np.zeros_like(records["true_increment"]),
                smooth, "by_gate",
            ),
        }
        # Derivative-free one-step flow comparison: the model's own next decoded
        # state against the exact HH relaxation step from the same decoded state.
        mapped_t = chart.predict(records["features"])
        mapped_next = chart.predict(records["features_next"])
        steady, tau = hh_kinetics(records["voltage"])
        hh_next = steady + (mapped_t - steady) * np.exp(-dt / tau)
        block["step_flow_state"] = score(hh_next, mapped_next, smooth, "by_gate")
        block["step_flow_increment"] = score(
            hh_next - mapped_t, mapped_next - mapped_t, smooth, "by_gate"
        )
        block["step_flow_persistence"] = score(
            hh_next - mapped_t, np.zeros_like(hh_next), smooth, "by_gate"
        )
        if field_analytic is not None:
            block["field_from_analytic_velocity"] = score(
                true_field, field_analytic, smooth, "by_gate"
            )
        blocks[f"stride_{stride}"] = block
        del validation_items, test_items, records
    return {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "evaluation_data": str(evaluation_root),
        "seed": args.seed,
        "windows_per_mode": args.windows_per_mode,
        "ridge_alpha": args.ridge_alpha,
        "chart_degree": args.chart_degree,
        "jvp_epsilon_ms": args.jvp_epsilon_ms,
        "latent_source": "observed history encoding followed by the model's own free rollout",
        "field_definition": (
            "chart chain rule along (dz/dt, dV/dt) with dz/dt from the model's own "
            "one-step map; identical chart, support, and metric as the transported-field audit"
        ),
        "blocks": blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evaluation-data", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19800)
    parser.add_argument("--windows-per-mode", type=int, default=200)
    parser.add_argument("--native-stride", type=int, default=1)
    parser.add_argument("--published-stride", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--chart-degree", type=int, default=3)
    parser.add_argument("--jvp-epsilon-ms", type=float, default=0.01)
    parser.add_argument("--smooth-slope-limit", type=float, default=1.0)
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
