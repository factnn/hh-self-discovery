#!/usr/bin/env python3
"""Verify that the Latent ODE pointwise field equals the model's own dynamics.

``ControlledLatentODE`` rolls out with ``torchdiffeq`` Euler on the grid
``t_k = k * dt_ms``, so for every sampled window the solver satisfies

    state_{k+1} - state_k = dt * f(t_k+1, state_k, u_{k+1})

where ``f`` is ``model.ode_func``.  This script checks that
``audit_latent_ode_chain.latent_ode_field`` reproduces those increments, which
is what licenses the transported-field audit to evaluate the field pointwise
(without integrating) at each rollout sample.

Two control-input pairings are reported:

* ``command_next`` -- ``u_{k+1}``, the command the solver actually used for the
  step, i.e. the exact identity (residual = float32 accumulation only);
* ``command_same`` -- ``u_k``, the command observed at the same time index as the
  state, which is the pairing used by the audit's flattened sample arrays.

Also reported: the sensitivity of the field to flipping the mode indicator, to
show that the per-mode ``set_context`` call matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_latent_ode_chain import MODES, latent_ode_field, normalize  # noqa: E402

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint  # noqa: E402


def _deviation(field: np.ndarray, increment: np.ndarray) -> dict:
    deviation = np.abs(field - increment)
    return {
        "mean_abs_deviation": float(deviation.mean()),
        "median_abs_deviation": float(np.median(deviation)),
        "p99_abs_deviation": float(np.quantile(deviation, 0.99)),
        "max_abs_deviation": float(deviation.max()),
        "mean_abs_increment": float(np.abs(increment).mean()),
        "relative_mean_abs_deviation": float(
            deviation.mean() / (np.abs(increment).mean() + 1e-12)
        ),
    }


def check_checkpoint(path: str, roots: dict[str, str], device, windows: int, seed: int) -> dict:
    model, checkpoint = load_latent_checkpoint(path, device)
    checkpoint = dict(checkpoint)
    checkpoint["_audit_stride"] = 1
    dt = checkpoint["dt_ms"]
    steps = checkpoint["prediction_steps"]
    output = {}
    for name, root in roots.items():
        items = _collect_windows(
            model, checkpoint, Path(root), "test", device, windows, seed, stride=1,
            strict_sampling=(name == "id"),
        )
        for mode in MODES:
            z_list, x_list, command_list = [], [], []
            for item in items:
                if item["mode"] != mode:
                    continue
                rows, latent_size = item["latents"].shape
                window_count = rows // steps
                z_list.append(item["latents"].reshape(window_count, steps, latent_size))
                x_list.append(item["prediction"].reshape(window_count, steps))
                command = item["current"] if mode == "current_clamp" else item["voltage"]
                command_list.append(command.reshape(window_count, steps))
            z = np.concatenate(z_list)
            x = np.concatenate(x_list)
            command = np.concatenate(command_list)
            window_count, steps, latent_size = z.shape
            block = np.repeat(mode, window_count * (steps - 1))
            start = z[:, :-1].reshape(-1, latent_size)
            increment = (z[:, 1:].reshape(-1, latent_size) - start) / dt
            state_response = normalize(x[:, :-1].reshape(-1), block, checkpoint, "response")
            same = normalize(command[:, :-1].reshape(-1), block, checkpoint, "command")
            nxt = normalize(command[:, 1:].reshape(-1), block, checkpoint, "command")
            field_same = latent_ode_field(model, start, state_response, same, block, device)
            field_next = latent_ode_field(model, start, state_response, nxt, block, device)
            flipped_mode = "voltage_clamp" if mode == "current_clamp" else "current_clamp"
            field_flipped = latent_ode_field(model, start, state_response, nxt, np.repeat(flipped_mode, len(block)), device)
            output[f"{name}/{mode}"] = {
                "samples": int(len(increment)),
                "windows": int(window_count),
                "command_next": _deviation(field_next, increment),
                "command_same": _deviation(field_same, increment),
                "mean_abs_field_change_when_mode_indicator_flipped": float(
                    np.abs(field_flipped - field_next).mean()
                ),
            }
    return {
        "checkpoint": path,
        "dt_ms": dt,
        "prediction_steps": steps,
        "windows_per_mode": windows,
        "seed": seed,
        "by_split_mode": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--windows-per-mode", type=int, default=20)
    parser.add_argument("--seed", type=int, default=19800)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "runs/baselines/b3_controlled_latent_ode_k2_s56/best.pt",
            "runs/baselines/b3_controlled_latent_ode_k2_s57/best.pt",
            "runs/baselines/b3_controlled_latent_ode_k2_s58/best.pt",
        ],
    )
    parser.add_argument("--output", default="runs/phase2/latent_ode_field_verification.json")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    roots = {
        "id": "data/generated/full_stratified_active_v1",
        "ood": "data/generated/ood_protocols",
    }
    result = {
        "definition": (
            "state_{k+1} - state_k is compared with dt * f(state_k, u) evaluated pointwise by "
            "audit_latent_ode_chain.latent_ode_field on the same free-rollout samples; "
            "'command_next' is the command the Euler step actually used, 'command_same' is the "
            "same-time-index command used by the flattened audit arrays."
        ),
        "per_checkpoint": [
            check_checkpoint(path, roots, device, args.windows_per_mode, args.seed)
            for path in args.checkpoints
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
