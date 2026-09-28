"""Cross-model latent alignment without reading evaluation-only HH gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import DataLoader

from .blind_evaluation import load_latent_checkpoint
from .sampling import BalancedWindowSampler, ObservedDataset, WindowReference
from .training import ModeNormalizers, Standardizer, WindowTorchDataset


def _normalizers(checkpoint: dict, mode: str) -> ModeNormalizers:
    values = checkpoint["normalizers"][mode]
    return ModeNormalizers(
        Standardizer(**values["command"]),
        Standardizer(**values["response"]),
    )


def shared_references(
    data_root: str | Path,
    split: str,
    history_steps: int,
    prediction_steps: int,
    windows_per_mode: int,
    seed: int,
) -> dict[str, list[WindowReference]]:
    """Choose one set of observable windows to use for every checkpoint."""
    dataset = ObservedDataset(data_root, split)
    sampler = BalancedWindowSampler(dataset, history_steps, prediction_steps)
    return {
        mode: sampler.sample_mode_epoch(mode, windows_per_mode, seed + index)
        for index, mode in enumerate(("current_clamp", "voltage_clamp"))
    }


def collect_latent_rollouts(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str,
    references: dict[str, list[WindowReference]],
    device_name: str = "cuda",
    stride: int = 5,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """Collect aligned latent states, vector fields, and observable predictions."""
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    dataset = ObservedDataset(data_root, split)
    sampler = BalancedWindowSampler(
        dataset, checkpoint["history_steps"], checkpoint["prediction_steps"]
    )
    states, fields, predictions, targets, modes, groups = [], [], [], [], [], []
    time_slice = slice(None, None, stride)
    group_offset = 0

    for mode_index, mode in enumerate(model.MODES):
        refs = references[mode]
        normalizers = _normalizers(checkpoint, mode)
        loader = DataLoader(
            WindowTorchDataset(sampler, refs, normalizers),
            batch_size=batch_size,
            shuffle=False,
        )
        window_offset = 0
        with torch.no_grad():
            for batch in loader:
                history = batch["history"].to(device)
                command = batch["future_command"].to(device)
                prediction, latent = model(history, command, mode, return_latents=True)
                target = normalizers.response.inverse(batch["target"].numpy())
                prediction_physical = normalizers.response.inverse(prediction.cpu().numpy())
                if mode == "current_clamp":
                    voltage = torch.as_tensor(target, dtype=torch.float32, device=device)
                else:
                    voltage = model.denormalize(command, mode, "command")
                steady, tau_ms = model.gate_kinetics(voltage)
                vector_field = (steady - latent) / tau_ms

                batch_size_actual, time_steps, latent_size = latent.shape
                states.append(
                    latent[:, time_slice].cpu().numpy().reshape(-1, latent_size)
                )
                fields.append(
                    vector_field[:, time_slice].cpu().numpy().reshape(-1, latent_size)
                )
                predictions.append(prediction_physical[:, time_slice].reshape(-1))
                targets.append(target[:, time_slice].reshape(-1))
                modes.append(
                    np.full(
                        batch_size_actual * len(range(0, time_steps, stride)),
                        mode_index,
                        dtype=np.int8,
                    )
                )
                sampled_steps = len(range(0, time_steps, stride))
                groups.append(
                    np.repeat(
                        np.arange(window_offset, window_offset + batch_size_actual) + group_offset,
                        sampled_steps,
                    )
                )
                window_offset += batch_size_actual
        group_offset += len(refs)
    return {
        "state": np.concatenate(states),
        "field": np.concatenate(fields),
        "prediction": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "group": np.concatenate(groups),
        "mode": np.concatenate(modes),
    }


@dataclass
class FittedMap:
    name: str
    predict: Callable[[np.ndarray], np.ndarray]


def fit_map(name: str, source: np.ndarray, target: np.ndarray) -> FittedMap:
    """Fit a controlled low-complexity map from source to target coordinates."""
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must have identical [sample, state] shapes")
    if name == "orthogonal":
        source_mean, target_mean = source.mean(0), target.mean(0)
        rotation, scale = orthogonal_procrustes(source - source_mean, target - target_mean)
        factor = scale / np.square(source - source_mean).sum()

        def predict(values: np.ndarray) -> np.ndarray:
            return (values - source_mean) @ rotation * factor + target_mean

        return FittedMap(name, predict)
    if name == "affine":
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=1e-6))
    elif name.startswith("polynomial"):
        degree = int(name.removeprefix("polynomial"))
        estimator = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            Ridge(alpha=1e-3),
        )
    else:
        raise ValueError(f"unknown map family: {name}")
    estimator.fit(source, target)
    return FittedMap(name, estimator.predict)


def _r2_by_dimension(target: np.ndarray, prediction: np.ndarray) -> list[float]:
    return [float(r2_score(target[:, i], prediction[:, i])) for i in range(target.shape[1])]


def _directional_pushforward(
    mapping: FittedMap,
    state: np.ndarray,
    vector_field: np.ndarray,
    epsilon: float = 1e-4,
) -> np.ndarray:
    plus = mapping.predict(state + epsilon * vector_field)
    minus = mapping.predict(state - epsilon * vector_field)
    return (plus - minus) / (2.0 * epsilon)


def evaluate_map(
    forward: FittedMap,
    reverse: FittedMap,
    source_state: np.ndarray,
    target_state: np.ndarray,
    source_field: np.ndarray,
    target_field: np.ndarray,
) -> dict:
    mapped = forward.predict(source_state)
    cycled = reverse.predict(mapped)
    pushed_field = _directional_pushforward(forward, source_state, source_field)
    field_scale = float(np.sqrt(np.mean(np.square(target_field))) + 1e-12)
    field_error = float(np.sqrt(np.mean(np.square(pushed_field - target_field))))
    flat_pushed, flat_target = pushed_field.reshape(-1), target_field.reshape(-1)
    cosine = float(
        np.dot(flat_pushed, flat_target)
        / ((np.linalg.norm(flat_pushed) * np.linalg.norm(flat_target)) + 1e-12)
    )
    return {
        "test_r2": _r2_by_dimension(target_state, mapped),
        "mean_test_r2": float(r2_score(target_state, mapped, multioutput="variance_weighted")),
        "cycle_r2": _r2_by_dimension(source_state, cycled),
        "mean_cycle_r2": float(r2_score(source_state, cycled, multioutput="variance_weighted")),
        "vector_field_relative_rmse": field_error / field_scale,
        "vector_field_cosine": cosine,
    }


def _bootstrap_output_disagreement(
    difference: np.ndarray,
    target: np.ndarray,
    group: np.ndarray,
    seed: int,
    samples: int = 1000,
) -> dict:
    unique_groups, inverse = np.unique(group, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    difference_second = np.bincount(inverse, weights=np.square(difference)) / count
    target_first = np.bincount(inverse, weights=target) / count
    target_second = np.bincount(inverse, weights=np.square(target)) / count

    def statistic(indices: np.ndarray) -> float:
        mse = float(difference_second[indices].mean())
        mean = float(target_first[indices].mean())
        variance = float(target_second[indices].mean() - mean * mean)
        return float(np.sqrt(mse) / (np.sqrt(max(variance, 0.0)) + 1e-12))

    observed = statistic(np.arange(len(unique_groups)))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(unique_groups), size=len(unique_groups))
        bootstrap[index] = statistic(selected)
    return {
        "normalized_rmse": observed,
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "windows": int(len(unique_groups)),
    }


def output_equivalence(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    seed: int = 811,
    equivalence_bound: float = 0.1,
) -> dict:
    """Compare predictions in each physical mode using window bootstrap CIs."""
    by_mode = {}
    for mode_index, mode_name in enumerate(("current_clamp", "voltage_clamp")):
        mask = source["mode"] == mode_index
        difference = source["prediction"][mask] - target["prediction"][mask]
        observed = source["target"][mask]
        entry = _bootstrap_output_disagreement(
            difference,
            observed,
            source["group"][mask],
            seed + mode_index,
        )
        source_error = source["prediction"][mask] - observed
        target_error = target["prediction"][mask] - observed
        scale = float(np.std(observed) + 1e-12)
        entry.update({
            "source_to_observation_nrmse": float(np.sqrt(np.mean(source_error**2)) / scale),
            "target_to_observation_nrmse": float(np.sqrt(np.mean(target_error**2)) / scale),
            "equivalence_bound": equivalence_bound,
            "passes_equivalence_bound": bool(entry["bootstrap_95_ci"][1] <= equivalence_bound),
        })
        by_mode[mode_name] = entry
    return {
        "by_mode": by_mode,
        "all_modes_equivalent": all(
            entry["passes_equivalence_bound"] for entry in by_mode.values()
        ),
    }


def align_pair(
    validation_source: dict[str, np.ndarray],
    validation_target: dict[str, np.ndarray],
    test_source: dict[str, np.ndarray],
    test_target: dict[str, np.ndarray],
    families: tuple[str, ...] = ("orthogonal", "affine", "polynomial2", "polynomial3"),
    bootstrap_seed: int = 811,
) -> dict:
    """Fit maps on validation and evaluate coordinates and dynamics on test."""
    output_disagreement = test_source["prediction"] - test_target["prediction"]
    target_scale = float(np.std(test_source["target"]) + 1e-12)
    result = {
        "output_disagreement_rmse": float(np.sqrt(np.mean(np.square(output_disagreement)))),
        "output_disagreement_normalized_rmse": float(
            np.sqrt(np.mean(np.square(output_disagreement))) / target_scale
        ),
        "output_equivalence": output_equivalence(
            test_source, test_target, seed=bootstrap_seed
        ),
        "maps": {},
    }
    for family in families:
        forward = fit_map(family, validation_source["state"], validation_target["state"])
        reverse = fit_map(family, validation_target["state"], validation_source["state"])
        result["maps"][family] = evaluate_map(
            forward,
            reverse,
            test_source["state"],
            test_target["state"],
            test_source["field"],
            test_target["field"],
        )
    return result


def affine_positive_control(
    validation_state: np.ndarray,
    test_state: np.ndarray,
    validation_field: np.ndarray,
    test_field: np.ndarray,
    seed: int = 1729,
) -> dict:
    """Verify that the audit recognizes a known smooth coordinate change."""
    rng = np.random.default_rng(seed)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(validation_state.shape[1],) * 2))
    scales = np.linspace(0.7, 1.4, validation_state.shape[1])
    matrix = orthogonal @ np.diag(scales)
    offset = rng.uniform(-0.2, 0.2, size=validation_state.shape[1])
    transformed_validation = validation_state @ matrix + offset
    transformed_test = test_state @ matrix + offset
    transformed_validation_field = validation_field @ matrix
    transformed_test_field = test_field @ matrix
    source_validation = {"state": transformed_validation}
    target_validation = {"state": validation_state}
    forward = fit_map("affine", source_validation["state"], target_validation["state"])
    reverse = fit_map("affine", target_validation["state"], source_validation["state"])
    return evaluate_map(
        forward,
        reverse,
        transformed_test,
        test_state,
        transformed_test_field,
        test_field,
    )
