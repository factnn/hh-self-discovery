#!/usr/bin/env python3
"""Transported-field audit for the controlled Latent ODE baseline (E198 protocol).

This is the continuous-dynamics-baseline counterpart of
``scripts/audit_claim_chain.py``.  It re-uses that script's protocol helpers
verbatim (window collection, cubic ridge chart, pushforward by central
differences, HH reference field, masks, metrics) and only swaps the definition
of the latent velocity:

* structured model:  ``dz/dt = (steady(V) - z) / tau(V)`` from ``gate_kinetics``
* Latent ODE:        ``dz/dt = f(z, x, u)`` from the neural vector field
  ``model.ode_func`` *evaluated pointwise at the free-rollout state* -- the ODE
  is never integrated here.

The Latent ODE state is ``[z (latent_size), x (normalized response scalar)]``
and the control input ``u`` is the normalized command (``V_cmd`` under voltage
clamp), so ``f`` is directly analogous to the structured relaxation.  Note that
unlike the structured model the field is *also* a function of the response
coordinate ``x`` carried inside the ODE state; both choices (the rollout's own
``x`` and the observed response) are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Re-use the locked E198 protocol exactly; the module has no import side effects.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_claim_chain import (  # noqa: E402
    GATES,
    closure_scores,
    field_scores,
    fit_chart,
    flatten,
    gate_scores,
    hh_field,
    jump_scores,
    prediction_scores,
    push_chart,
)

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint  # noqa: E402

FIELD_CHUNK = 65536
MODES = ("current_clamp", "voltage_clamp")


def extra_arrays(items: list[dict]) -> dict[str, np.ndarray]:
    """Per-sample arrays that ``flatten`` does not expose, in its exact order."""
    return {
        "model_response": np.concatenate([item["prediction"].reshape(-1) for item in items]),
        "observed_response": np.concatenate([item["target"].reshape(-1) for item in items]),
        "observed_command": np.concatenate([
            (item["current"] if item["mode"] == "current_clamp" else item["voltage"]).reshape(-1)
            for item in items
        ]),
    }


def normalize(values: np.ndarray, modes: np.ndarray, checkpoint: dict, key: str) -> np.ndarray:
    """Apply the per-mode standardizer (``command`` or ``response``) to physical values."""
    output = np.empty(len(values), dtype=np.float64)
    for mode in MODES:
        selected = modes == mode
        if not selected.any():
            continue
        stats = checkpoint["normalizers"][mode][key]
        output[selected] = (values[selected] - stats["mean"]) / stats["std"]
    return output


def latent_ode_field(
    model,
    z: np.ndarray,
    response_state: np.ndarray,
    command: np.ndarray,
    modes: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Evaluate ``dz/dt = ode_func(t, [z, x])`` at each point; no integration.

    ``response_state`` and ``command`` must already be normalized exactly as the
    rollout sees them.  The mode indicator supplied to ``set_context`` is the
    sample's own control mode.
    """
    velocity = np.empty_like(z, dtype=np.float64)
    with torch.no_grad():
        for mode in MODES:
            index = np.flatnonzero(modes == mode)
            if index.size == 0:
                continue
            for start in range(0, index.size, FIELD_CHUNK):
                chunk = index[start : start + FIELD_CHUNK]
                state = torch.as_tensor(
                    np.column_stack((z[chunk], response_state[chunk])),
                    dtype=torch.float32,
                    device=device,
                )
                control = torch.as_tensor(
                    command[chunk, None], dtype=torch.float32, device=device
                )
                model.ode_func.set_context(control, mode)
                time = torch.zeros(1, dtype=state.dtype, device=device)
                full = model.ode_func(time, state)
                velocity[chunk] = full[:, : model.latent_size].cpu().numpy()
    return velocity


def structured_field(model, z: np.ndarray, voltage: np.ndarray, device: torch.device) -> np.ndarray:
    """Reference implementation copied from ``audit_claim_chain.latent_field``."""
    parts = []
    with torch.no_grad():
        for start in range(0, len(z), FIELD_CHUNK):
            state = torch.as_tensor(z[start : start + FIELD_CHUNK], dtype=torch.float32, device=device)
            v = torch.as_tensor(voltage[start : start + FIELD_CHUNK], dtype=torch.float32, device=device)
            steady, tau = model.gate_kinetics(v)
            parts.append(((steady - state) / tau).cpu().numpy())
    return np.concatenate(parts)


def is_latent_ode(model) -> bool:
    return hasattr(model, "ode_func") and not hasattr(model, "gate_kinetics")


def latent_velocity(
    model,
    arrays: dict[str, np.ndarray],
    checkpoint: dict,
    response_variant: str,
    device: torch.device,
) -> np.ndarray:
    """Latent velocity field for either family, on the flattened sample order."""
    modes = arrays["mode"]
    if is_latent_ode(model):
        response = normalize(arrays[response_variant], modes, checkpoint, "response")
        command = normalize(arrays["observed_command"], modes, checkpoint, "command")
        return latent_ode_field(model, arrays["z"], response, command, modes, device)
    return structured_field(model, arrays["z"], arrays["true_v"], device)


def evaluate(args) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(args.checkpoint, device)
    checkpoint = dict(checkpoint)
    checkpoint["_audit_stride"] = args.stride
    root = Path(args.data)
    evaluation_root = Path(args.evaluation_data)
    validation_items = _collect_windows(
        model, checkpoint, root, "validation", device, args.windows_per_mode,
        args.seed, stride=args.stride,
    )
    test_items = _collect_windows(
        model, checkpoint, evaluation_root, "test", device, args.windows_per_mode,
        args.seed + 10000, stride=args.stride, strict_sampling=evaluation_root == root,
    )
    train = {**flatten(validation_items, checkpoint), **extra_arrays(validation_items)}
    test = {**flatten(test_items, checkpoint), **extra_arrays(test_items)}

    train_features = np.column_stack((train["z"], train["true_v"]))
    test_features = np.column_stack((test["z"], test["true_v"]))
    z_chart = fit_chart(train["z"], train["gates"], args.ridge_alpha)
    v_chart = fit_chart(train["true_v"][:, None], train["gates"], args.ridge_alpha)
    zv_chart = fit_chart(train_features, train["gates"], args.ridge_alpha)
    test_mapped = zv_chart.predict(test_features)
    z_only_mapped = z_chart.predict(test["z"])
    voltage_only_mapped = v_chart.predict(test["true_v"][:, None])

    train_latent_field = latent_velocity(
        model, train, checkpoint, args.response_variant, device
    )
    test_latent_field = latent_velocity(
        model, test, checkpoint, args.response_variant, device
    )
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
    if is_latent_ode(model):
        # The Latent ODE field is conditioned on the command and on the state's own
        # response coordinate, never on V; the primary branch already uses both, so
        # only the chart's voltage feature changes (observed V -> predicted V_hat).
        model_latent_field = test_latent_field
    else:
        model_latent_field = structured_field(model, test["z"], test["model_v"], device)
    model_mapped = zv_chart.predict(model_features)
    model_pushed = push_chart(
        zv_chart, model_features,
        np.column_stack((model_latent_field, test["model_vdot"])), args.jvp_epsilon_ms,
    )

    diagnostic_field = None
    diagnostic_pushed = None
    if is_latent_ode(model) and args.response_variant != "observed_response":
        diagnostic_field = latent_velocity(
            model, test, checkpoint, "observed_response", device
        )
        diagnostic_pushed = push_chart(
            zv_chart, test_features,
            np.column_stack((diagnostic_field, test["true_vdot"])), args.jvp_epsilon_ms,
        )

    # Decomposition controls: transport the identical chart with only one of the two
    # components of the push direction, to show how much of the field is driven by
    # the latent dynamics versus the imposed voltage command.
    nonzero_command_slope = np.abs(
        test["true_vdot"][(test["mode"] == "voltage_clamp") & (test["true_vdot"] != 0.0)]
    )
    pushed_voltage_only = push_chart(
        zv_chart, test_features,
        np.column_stack((np.zeros_like(test_latent_field), test["true_vdot"])),
        args.jvp_epsilon_ms,
    )
    pushed_latent_only = push_chart(
        zv_chart, test_features,
        np.column_stack((test_latent_field, np.zeros_like(test["true_vdot"]))),
        args.jvp_epsilon_ms,
    )

    masks = {
        "all": np.ones(len(test["z"]), dtype=bool),
        "current_clamp": test["mode"] == "current_clamp",
        "voltage_clamp": test["mode"] == "voltage_clamp",
        "voltage_clamp_smooth": (test["mode"] == "voltage_clamp") & (np.abs(test["true_vdot"]) <= 1.0),
    }
    transported = {
        "observed_or_command_voltage_local": {
            name: field_scores(true_field, test_pushed, mask) for name, mask in masks.items()
        },
        "rollout_or_command_voltage": {
            name: field_scores(true_field, model_pushed, mask) for name, mask in masks.items()
        },
        "field_definition": {
            "latent_velocity": (
                "model.ode_func(time, state=[z, response_state]) evaluated pointwise at the "
                "free-rollout state; no ODE integration"
                if is_latent_ode(model)
                else "(steady(V) - z) / tau(V)"
            ),
            "control_input": (
                "true observed command u(t) of the window, normalized with the mode command "
                "standardizer (V_cmd under voltage clamp)"
            ),
            "response_state": args.response_variant,
            "response_state_note": (
                "the Latent ODE field also depends on the normalized response coordinate "
                "carried inside the ODE state; 'model_response' is the rollout's own coordinate"
            ),
            "voltage_component": "observed dV/dt of the window (identical to the structured audit)",
        },
        "voltage_semantics": {
            "current_clamp": (
                "first branch uses observed response V; second uses freely predicted response V_hat"
            ),
            "voltage_clamp": "both branches use the same V_cmd and dV_cmd/dt",
        },
    }
    if diagnostic_pushed is not None:
        transported["observed_response_state_diagnostic"] = {
            "note": (
                "same chart and support; the ODE state's response coordinate is replaced by the "
                "normalized observed response instead of the rollout's own coordinate"
            ),
            "field_scores": {
                name: field_scores(true_field, diagnostic_pushed, mask) for name, mask in masks.items()
            },
        }
    transported["push_decomposition_controls"] = {
        "note": (
            "identical chart and support, but the push direction keeps only one component: "
            "'latent_only' pushes along (dz/dt, 0), 'voltage_only' along (0, dV_cmd/dt). "
            "On the smooth voltage-clamp support dV_cmd/dt == 0 exactly (see "
            "support_diagnostics), so 'voltage_only' is identically zero there and the "
            "transported field reduces to the latent pushforward."
        ),
        "latent_only": {
            name: field_scores(true_field, pushed_latent_only, mask) for name, mask in masks.items()
        },
        "voltage_only": {
            name: field_scores(true_field, pushed_voltage_only, mask) for name, mask in masks.items()
        },
    }

    return {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "evaluation_data": args.evaluation_data,
        "seed": args.seed,
        "windows_per_mode": args.windows_per_mode,
        "stride": args.stride,
        "ridge_alpha": args.ridge_alpha,
        "jvp_epsilon_ms": args.jvp_epsilon_ms,
        "latent_source": "one observed history encoding followed by free rollout",
        "local_audit": "free-roll z paired with observed V and observed dV/dt",
        "model_family": "controlled_latent_ode" if is_latent_ode(model) else "structured_latent_hh",
        "support_diagnostics": {
            "voltage_clamp_smooth": {
                "definition": "voltage-clamp samples with |dV_cmd/dt| <= 1 mV/ms",
                "count": int(masks["voltage_clamp_smooth"].sum()),
                "fraction_with_exactly_zero_command_slope": float(
                    np.mean(test["true_vdot"][masks["voltage_clamp_smooth"]] == 0.0)
                ),
                "min_abs_nonzero_command_slope_over_all_voltage_clamp": (
                    float(nonzero_command_slope.min()) if nonzero_command_slope.size else None
                ),
                "note": (
                    "the command is piecewise constant at the audited grid, so the "
                    "|dV_cmd/dt| <= 1 mV/ms mask selects exactly the held-command samples "
                    "(dV_cmd/dt == 0); the transported field on this support is therefore the "
                    "pure latent pushforward with the chart's voltage coordinate frozen"
                ),
            },
            "voltage_clamp_plain": {
                "count": int(masks["voltage_clamp"].sum()),
                "fraction_with_exactly_zero_command_slope": float(
                    np.mean(test["true_vdot"][masks["voltage_clamp"]] == 0.0)
                ),
            },
        },
        "latent_field_diagnostics": {
            "voltage_clamp_smooth": {
                "latent_dimension": int(test["z"].shape[1]),
                "latent_rms": float(np.sqrt(np.mean(test["z"][masks["voltage_clamp_smooth"]] ** 2))),
                "latent_velocity_rms_per_ms": float(
                    np.sqrt(np.mean(test_latent_field[masks["voltage_clamp_smooth"]] ** 2))
                ),
                "latent_velocity_mean_abs_per_ms": float(
                    np.mean(np.abs(test_latent_field[masks["voltage_clamp_smooth"]]))
                ),
                "latent_velocity_max_abs_per_ms": float(
                    np.max(np.abs(test_latent_field[masks["voltage_clamp_smooth"]]))
                ),
                "true_gate_field_rms_per_ms": float(
                    np.sqrt(np.mean(true_field[masks["voltage_clamp_smooth"]] ** 2))
                ),
                "note": (
                    "scale check: a near-zero transported-field R2 is only degenerate if the "
                    "latent velocity is itself near zero; compare these RMS values with "
                    "true_gate_field_rms_per_ms"
                ),
            }
        },
        "prediction_same_windows": prediction_scores(test_items, checkpoint),
        "state_same_windows": {
            "z_only_gate_r2": gate_scores(test["gates"], z_only_mapped),
            "voltage_only_gate_r2": gate_scores(test["gates"], voltage_only_mapped),
            "z_observed_or_command_voltage_gate_r2": gate_scores(test["gates"], test_mapped),
            "z_rollout_or_command_voltage_gate_r2": gate_scores(test["gates"], model_mapped),
            "by_mode": {
                mode: {
                    "z_only_gate_r2": gate_scores(
                        test["gates"][test["mode"] == mode], z_only_mapped[test["mode"] == mode]
                    ),
                    "voltage_only_gate_r2": gate_scores(
                        test["gates"][test["mode"] == mode], voltage_only_mapped[test["mode"] == mode]
                    ),
                    "z_observed_or_command_voltage_gate_r2": gate_scores(
                        test["gates"][test["mode"] == mode], test_mapped[test["mode"] == mode]
                    ),
                    "z_rollout_or_command_voltage_gate_r2": gate_scores(
                        test["gates"][test["mode"] == mode], model_mapped[test["mode"] == mode]
                    ),
                }
                for mode in MODES
            },
            "paired_support": {
                "voltage_clamp_smooth": {
                    "definition": "voltage-clamp samples with |dV_cmd/dt| <= 1 mV/ms",
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
                },
                "voltage_clamp_all": {
                    "definition": "all voltage-clamp samples (no smoothness restriction)",
                    "count": int(masks["voltage_clamp"].sum()),
                    "z_only_gate_r2": gate_scores(
                        test["gates"][masks["voltage_clamp"]], z_only_mapped[masks["voltage_clamp"]]
                    ),
                    "voltage_only_gate_r2": gate_scores(
                        test["gates"][masks["voltage_clamp"]],
                        voltage_only_mapped[masks["voltage_clamp"]],
                    ),
                    "z_command_voltage_gate_r2": gate_scores(
                        test["gates"][masks["voltage_clamp"]], test_mapped[masks["voltage_clamp"]]
                    ),
                },
            },
            "voltage_semantics": {
                "current_clamp": "observed response V versus freely predicted response V_hat",
                "voltage_clamp": "both branches use the identical imposed command V_cmd",
            },
        },
        "transported_field_same_windows": transported,
        "m_closure": closure_scores(
            train, test, zv_chart.predict(train_features), test_mapped,
            train_pushed, test_pushed, args.seed,
        ),
        "voltage_jump_continuity": jump_scores(test_items, checkpoint, z_chart, zv_chart),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="chart-fitting root (ID)")
    parser.add_argument("--evaluation-data", default=None, help="evaluation root; defaults to --data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19800)
    parser.add_argument("--windows-per-mode", type=int, default=200)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--jvp-epsilon-ms", type=float, default=0.01)
    parser.add_argument(
        "--response-variant",
        default="model_response",
        choices=("model_response", "observed_response"),
        help="Latent ODE state response coordinate used in the primary field branch",
    )
    args = parser.parse_args()
    if args.evaluation_data is None:
        args.evaluation_data = args.data
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
