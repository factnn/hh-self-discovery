"""Fit smooth post-selection maps using state and vector-field supervision."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

from hh_self_discovery.blind_evaluation import _collect_windows, load_latent_checkpoint
from hh_self_discovery.simulator import gate_derivatives


class SmoothMap(torch.nn.Module):
    def __init__(self, latent_size, width=32, depth=2):
        super().__init__()
        layers = [torch.nn.Linear(latent_size, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([torch.nn.Linear(width, width), torch.nn.Tanh()])
        layers.extend([torch.nn.Linear(width, 3), torch.nn.Sigmoid()])
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values)


def torch_hh_gate_derivatives(voltage, gates):
    def vtrap(x, scale):
        ratio = x / scale
        denominator = -torch.expm1(-ratio)
        safe_denominator = torch.where(torch.abs(ratio) < 1e-4, torch.ones_like(ratio), denominator)
        regular = x / safe_denominator
        series = scale * (1.0 + ratio / 2.0 + ratio * ratio / 12.0)
        return torch.where(torch.abs(ratio) < 1e-4, series, regular)

    alpha_n = 0.01 * vtrap(voltage + 55.0, 10.0)
    beta_n = 0.125 * torch.exp(torch.clamp(-(voltage + 65.0) / 80.0, -80, 80))
    alpha_m = 0.1 * vtrap(voltage + 40.0, 10.0)
    beta_m = 4.0 * torch.exp(torch.clamp(-(voltage + 65.0) / 18.0, -80, 80))
    alpha_h = 0.07 * torch.exp(torch.clamp(-(voltage + 65.0) / 20.0, -80, 80))
    beta_h = 1.0 / (1.0 + torch.exp(torch.clamp(-(voltage + 35.0) / 10.0, -80, 80)))
    alpha = torch.stack((alpha_n, alpha_m, alpha_h), dim=-1)
    beta = torch.stack((beta_n, beta_m, beta_h), dim=-1)
    return alpha * (1.0 - gates) - beta * gates


def arrays(model, checkpoint, items, device, limit, seed, include_voltage):
    z = np.concatenate([item["latents"] for item in items])
    gates = np.concatenate([item["gates"] for item in items])
    voltage = np.concatenate([item["voltage"] for item in items])
    steps = checkpoint["prediction_steps"] // 5
    voltage_derivative = np.concatenate([
        np.gradient(item["voltage"].reshape(-1, steps), checkpoint["dt_ms"] * 5, axis=1).reshape(-1)
        for item in items
    ])
    sample_dt_ms = checkpoint["dt_ms"] * 5
    jump_distance_ms = []
    for item in items:
        voltage_rows = item["voltage"].reshape(-1, steps)
        for row in voltage_rows:
            jump_indices = np.flatnonzero(np.abs(np.diff(row)) >= 1.0) + 1
            if len(jump_indices) == 0:
                jump_distance_ms.append(np.full(steps, np.inf, dtype=np.float32))
                continue
            indices = np.arange(steps)
            distances = np.min(
                np.abs(indices[:, None] - jump_indices[None, :]), axis=1
            )
            jump_distance_ms.append((distances * sample_dt_ms).astype(np.float32))
    jump_distance_ms = np.concatenate(jump_distance_ms)
    modes = np.concatenate([
        np.repeat(item["mode"], len(item["latents"])) for item in items
    ])
    trajectory_ids = np.concatenate([item["trajectory_ids"] for item in items])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(z), size=min(limit, len(z)), replace=False)
    z, gates, voltage, modes = z[chosen], gates[chosen], voltage[chosen], modes[chosen]
    trajectory_ids = trajectory_ids[chosen]
    voltage_derivative = voltage_derivative[chosen]
    jump_distance_ms = jump_distance_ms[chosen]
    with torch.no_grad():
        v = torch.as_tensor(voltage, dtype=torch.float32, device=device)
        state = torch.as_tensor(z, dtype=torch.float32, device=device)
        steady, tau = model.gate_kinetics(v)
        latent_field = ((steady - state) / tau).cpu().numpy()
    true_field = np.stack([gate_derivatives(vv, gg) for vv, gg in zip(voltage, gates)])
    if include_voltage:
        z = np.column_stack([z, voltage])
        latent_field = np.column_stack([latent_field, voltage_derivative])
    strata = {
        "mode": modes,
        "smooth_dvdt_le_5": np.abs(voltage_derivative) <= 5.0,
        "smooth_dvdt_le_1": np.abs(voltage_derivative) <= 1.0,
        "jump_distance_ms": jump_distance_ms,
        "trajectory_ids": trajectory_ids,
    }
    return (z.astype("float32"), gates.astype("float32"),
            latent_field.astype("float32"), true_field.astype("float32"), strata)


def _metrics(mapped, pushed, gates, true_field):
    scale = np.sqrt(np.mean(true_field**2)) + 1e-12
    cosine = np.sum(pushed * true_field, axis=1) / (
        np.linalg.norm(pushed, axis=1) * np.linalg.norm(true_field, axis=1) + 1e-12
    )
    result = {
        "count": int(len(gates)),
        "mean_gate_r2": float(r2_score(gates, mapped, multioutput="variance_weighted")),
        "gate_r2": {gate: float(r2_score(gates[:, index], mapped[:, index]))
                    for index, gate in enumerate(("n", "m", "h"))},
        "field_relative_rmse": float(np.sqrt(np.mean((pushed - true_field) ** 2)) / scale),
        "field_mean_cosine": float(np.mean(cosine)),
        "field_positive_fraction": float(np.mean(cosine > 0)),
    }
    result["per_gate_field"] = {}
    for index, gate in enumerate(("n", "m", "h")):
        gate_scale = np.sqrt(np.mean(true_field[:, index] ** 2)) + 1e-12
        predicted = pushed[:, index]
        target = true_field[:, index]
        result["per_gate_field"][gate] = {
            "relative_rmse": float(np.sqrt(np.mean((predicted - target) ** 2)) / gate_scale),
            "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(target))),
            "correlation": float(np.corrcoef(predicted, target)[0, 1]),
        }
    return result


def _field_error_decomposition(pushed, hh_from_mapped, true_field):
    result = {}
    for index, gate in enumerate(("n", "m", "h")):
        scale = np.sqrt(np.mean(true_field[:, index] ** 2)) + 1e-12
        state_error = hh_from_mapped[:, index] - true_field[:, index]
        coordinate_error = pushed[:, index] - hh_from_mapped[:, index]
        total_error = pushed[:, index] - true_field[:, index]
        result[gate] = {
            "state_amplification_rmse": float(np.sqrt(np.mean(state_error**2)) / scale),
            "coordinate_residual_rmse": float(np.sqrt(np.mean(coordinate_error**2)) / scale),
            "total_rmse": float(np.sqrt(np.mean(total_error**2)) / scale),
            "error_correlation": float(np.corrcoef(state_error, coordinate_error)[0, 1]),
            "cross_term": float(2 * np.mean(state_error * coordinate_error) / (scale**2)),
        }
    return result


def score(mapping, z, gates, latent_field, true_field, strata, z_mean, z_std, device,
          epsilon, include_voltage):
    mapping.eval()
    batch = 4096
    mapped_parts, field_parts = [], []
    with torch.no_grad():
        for start in range(0, len(z), batch):
            state = torch.as_tensor(z[start:start + batch], device=device)
            field = torch.as_tensor(latent_field[start:start + batch], device=device)
            normalized = (state - z_mean) / z_std
            mapped = mapping(normalized)
            pushed = (
                mapping((state + epsilon * field - z_mean) / z_std)
                - mapping((state - epsilon * field - z_mean) / z_std)
            ) / (2 * epsilon)
            mapped_parts.append(mapped.cpu().numpy())
            field_parts.append(pushed.cpu().numpy())
    mapped = np.concatenate(mapped_parts)
    pushed = np.concatenate(field_parts)
    result = _metrics(mapped, pushed, gates, true_field)
    if include_voltage:
        hh_from_mapped = np.stack([
            gate_derivatives(voltage, gate_state)
            for voltage, gate_state in zip(z[:, -1], mapped)
        ])
        result["hh_field_from_mapped_state"] = _metrics(
            mapped, hh_from_mapped, gates, true_field
        )
        result["field_error_decomposition"] = _field_error_decomposition(
            pushed, hh_from_mapped, true_field
        )
    result["stratified"] = {}
    for mode in np.unique(strata["mode"]):
        mode_mask = strata["mode"] == mode
        result["stratified"][mode] = {
            "all": _metrics(mapped[mode_mask], pushed[mode_mask], gates[mode_mask], true_field[mode_mask])
        }
        if mode == "voltage_clamp":
            for name in ("smooth_dvdt_le_5", "smooth_dvdt_le_1"):
                mask = mode_mask & strata[name]
                result["stratified"][mode][name] = _metrics(
                    mapped[mask], pushed[mask], gates[mask], true_field[mask])
            event_bins = {
                "jump_distance_le_0p5ms": strata["jump_distance_ms"] <= 0.5,
                "jump_distance_0p5_to_2ms": (
                    (strata["jump_distance_ms"] > 0.5)
                    & (strata["jump_distance_ms"] <= 2.0)
                ),
                "jump_distance_gt_2ms": strata["jump_distance_ms"] > 2.0,
            }
            for name, event_mask in event_bins.items():
                mask = mode_mask & event_mask
                if np.any(mask):
                    entry = _metrics(
                        mapped[mask], pushed[mask], gates[mask], true_field[mask]
                    )
                    if include_voltage:
                        entry["hh_field_from_mapped_state"] = _metrics(
                            mapped[mask], hh_from_mapped[mask], gates[mask], true_field[mask]
                        )
                        entry["field_error_decomposition"] = _field_error_decomposition(
                            pushed[mask], hh_from_mapped[mask], true_field[mask]
                        )
                    result["stratified"][mode][name] = entry
    return result


def evaluate(
    checkpoint_path, data_root, device_name, seed, width, depth, steps, weights,
    learning_rate, batch_size, grad_clip, include_voltage, mode, jvp_epsilon,
    field_target, state_gate_weights, select_best
    , calibration_split, calibration_field_objective
):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    root = Path(data_root)
    validation_items = _collect_windows(model, checkpoint, root, "validation", device, 300, seed)
    test_items = _collect_windows(model, checkpoint, root, "test", device, 300, seed + 10000)
    if mode != "all":
        validation_items = [item for item in validation_items if item["mode"] == mode]
        test_items = [item for item in test_items if item["mode"] == mode]
    validation = arrays(
        model, checkpoint, validation_items, device, 80000, seed + 1, include_voltage
    )
    test = arrays(
        model, checkpoint, test_items, device, 80000, seed + 2, include_voltage
    )
    z_mean = torch.as_tensor(validation[0].mean(0), device=device)
    z_std = torch.as_tensor(validation[0].std(0) + 1e-6, device=device)
    gate_std = torch.as_tensor(validation[1].std(0) + 1e-6, device=device)
    state_weights = torch.as_tensor(state_gate_weights, dtype=torch.float32, device=device)
    field_rms = torch.as_tensor(
        np.sqrt(np.mean(validation[3] ** 2, axis=0)) + 1e-6, device=device
    )
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(validation[0]))
    if calibration_split == "trajectory":
        unique_ids = np.unique(validation[4]["trajectory_ids"])
        calibration_ids = rng.choice(
            unique_ids, size=max(1, len(unique_ids) // 5), replace=False
        )
        candidates = all_indices[np.isin(validation[4]["trajectory_ids"], calibration_ids)]
        calibration_indices = rng.choice(
            candidates, size=min(8192, len(candidates)), replace=False
        )
        fit_indices = all_indices[
            ~np.isin(validation[4]["trajectory_ids"], calibration_ids)
        ] if select_best else all_indices
    else:
        calibration_indices = rng.choice(
            all_indices, size=min(8192, len(all_indices) // 10), replace=False
        )
        fit_indices = np.setdiff1d(all_indices, calibration_indices) if select_best else all_indices
    calibration_far_mask = (
        (validation[4]["mode"][calibration_indices] == "voltage_clamp")
        & (validation[4]["jump_distance_ms"][calibration_indices] > 2.0)
    )
    if calibration_field_objective in ("true_field_far", "true_m_field_far") and not np.any(calibration_far_mask):
        raise ValueError("No far-support voltage-clamp samples in calibration split")
    results = {}
    for weight in weights:
        torch.manual_seed(seed + int(weight * 1000))
        mapping = SmoothMap(validation[0].shape[1], width, depth).to(device)
        optimizer = torch.optim.AdamW(
            mapping.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        history = []
        best_objective = float("inf")
        best_state = None
        for step in range(steps):
            indices = rng.choice(fit_indices, size=batch_size, replace=True)
            state = torch.as_tensor(validation[0][indices], device=device)
            gates = torch.as_tensor(validation[1][indices], device=device)
            latent_field = torch.as_tensor(validation[2][indices], device=device)
            true_field = torch.as_tensor(validation[3][indices], device=device)
            normalized = (state - z_mean) / z_std
            mapped = mapping(normalized)
            state_loss = torch.mean(((mapped - gates) / gate_std) ** 2 * state_weights)
            epsilon = jvp_epsilon
            pushed = (
                mapping((state + epsilon * latent_field - z_mean) / z_std)
                - mapping((state - epsilon * latent_field - z_mean) / z_std)
            ) / (2 * epsilon)
            target_field = true_field
            if field_target == "mapped_hh":
                target_field = torch_hh_gate_derivatives(state[:, -1], mapped)
            field_loss = torch.mean(((pushed - target_field) / field_rms) ** 2)
            loss = state_loss + weight * field_loss
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(mapping.parameters(), grad_clip)
            optimizer.step()
            if step % 500 == 0 or step == steps - 1:
                record = {
                    "step": step,
                    "state_loss": float(state_loss.detach()),
                    "field_loss": float(field_loss.detach()),
                }
                if select_best:
                    with torch.no_grad():
                        ci = calibration_indices
                        cstate = torch.as_tensor(validation[0][ci], device=device)
                        cgates = torch.as_tensor(validation[1][ci], device=device)
                        clatent_field = torch.as_tensor(validation[2][ci], device=device)
                        cmapped = mapping((cstate - z_mean) / z_std)
                        cpushed = (
                            mapping((cstate + jvp_epsilon * clatent_field - z_mean) / z_std)
                            - mapping((cstate - jvp_epsilon * clatent_field - z_mean) / z_std)
                        ) / (2 * jvp_epsilon)
                        ctrue = torch.as_tensor(validation[3][ci], device=device)
                        ctarget = ctrue
                        if field_target == "mapped_hh":
                            ctarget = torch_hh_gate_derivatives(cstate[:, -1], cmapped)
                        cstate_loss = torch.mean(((cmapped - cgates) / gate_std) ** 2 * state_weights)
                        if calibration_field_objective == "true_field_far":
                            cfar = torch.as_tensor(calibration_far_mask, device=device)
                            cfield_loss = torch.mean(
                                ((cpushed[cfar] - ctrue[cfar]) / field_rms) ** 2
                            )
                        elif calibration_field_objective == "true_m_field_far":
                            cfar = torch.as_tensor(calibration_far_mask, device=device)
                            cfield_loss = torch.mean(
                                ((cpushed[cfar, 1] - ctrue[cfar, 1]) / field_rms[1]) ** 2
                            )
                        else:
                            cfield_loss = torch.mean(((cpushed - ctarget) / field_rms) ** 2)
                        objective = float(cstate_loss + weight * cfield_loss)
                    record["calibration_objective"] = objective
                    if objective < best_objective:
                        best_objective = objective
                        best_state = {
                            name: value.detach().cpu().clone()
                            for name, value in mapping.state_dict().items()
                        }
                history.append(record)
        if best_state is not None:
            mapping.load_state_dict(best_state)
        results[f"field_weight_{weight:g}"] = {
            "history": history,
            "best_calibration_objective": best_objective if select_best else None,
            "test": score(
                mapping, *test, z_mean, z_std, device, jvp_epsilon, include_voltage
            ),
        }
    return {
        "checkpoint": checkpoint_path,
        "validation_samples": len(validation[0]),
        "test_samples": len(test[0]),
        "steps_per_weight": steps,
        "width": width,
        "depth": depth,
        "field_weights": weights,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "grad_clip": grad_clip,
        "include_voltage": include_voltage,
        "mode": mode,
        "jvp_epsilon": jvp_epsilon,
        "field_target": field_target,
        "state_gate_weights": state_gate_weights,
        "select_best": select_best,
        "calibration_split": calibration_split,
        "calibration_field_objective": calibration_field_objective,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=5656)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--field-weights", default="0,0.01,0.1,1")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--include-voltage", action="store_true")
    parser.add_argument(
        "--mode", choices=("all", "current_clamp", "voltage_clamp"), default="all"
    )
    parser.add_argument("--jvp-epsilon", type=float, default=1e-2)
    parser.add_argument("--field-target", choices=("true", "mapped_hh"), default="true")
    parser.add_argument("--state-gate-weights", default="1,1,1")
    parser.add_argument("--select-best", action="store_true")
    parser.add_argument("--calibration-split", choices=("point", "trajectory"), default="point")
    parser.add_argument(
        "--calibration-field-objective",
        choices=("training_target", "true_field_far", "true_m_field_far"),
        default="training_target",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    weights = [float(value) for value in args.field_weights.split(",")]
    state_gate_weights = [float(value) for value in args.state_gate_weights.split(",")]
    if len(state_gate_weights) != 3:
        parser.error("--state-gate-weights requires n,m,h weights")
    result = evaluate(
        args.checkpoint, args.data, args.device, args.seed,
        args.width, args.depth, args.steps, weights,
        args.learning_rate, args.batch_size, args.grad_clip, args.include_voltage,
        args.mode, args.jvp_epsilon, args.field_target, state_gate_weights
        , args.select_best, args.calibration_split, args.calibration_field_objective
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
