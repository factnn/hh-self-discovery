"""Post-selection evaluation that is the only code allowed to read true gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import DataLoader

from .latent_model import StructuredLatentHH
from .metrics import regression_metrics, waveform_metrics
from .sampling import BalancedWindowSampler, ObservedDataset
from .training import ModeNormalizers, Standardizer, WindowTorchDataset


TRUE_GATES = ("n", "m", "h")


def _lag_features(values: np.ndarray, lags: tuple[int, ...] = (1, 5, 10, 20, 50)) -> np.ndarray:
    """Return padded lagged columns for a batch of subsampled time series."""
    shifted_values = []
    for lag in lags:
        shifted = np.empty_like(values)
        shifted[:, :lag] = values[:, :1]
        shifted[:, lag:] = values[:, :-lag]
        shifted_values.append(shifted)
    return np.stack(shifted_values, axis=-1).reshape(-1, len(lags))


def _history_summary(values: np.ndarray) -> np.ndarray:
    """Compact multiscale summary of the encoder input history."""
    lookbacks = (1, 5, 25, 100, 250, 500)
    sampled = [values[:, -min(lag, values.shape[1])] for lag in lookbacks]
    return np.column_stack(sampled + [
        values.mean(axis=1), values.std(axis=1), values.min(axis=1), values.max(axis=1)
    ])


def _history_grid(values: np.ndarray, points: int = 32) -> np.ndarray:
    """Uniformly sample the raw encoder history without using hidden state."""
    indices = np.linspace(0, values.shape[1] - 1, points).round().astype(int)
    return values[:, indices]


def load_latent_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device)
    if "config" in checkpoint and "model_type" in checkpoint["config"]:
        from .baseline_models import build_baseline_model

        config = checkpoint["config"]
        model = build_baseline_model(
            config["model_type"],
            config["latent_size"],
            encoder_width=config["encoder_width"],
            dynamics_width=config["dynamics_width"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        checkpoint = {
            **checkpoint,
            "latent_size": config["latent_size"],
            "history_steps": config["history_steps"],
            "prediction_steps": config["prediction_steps"],
            "dt_ms": 0.02,
        }
        return model, checkpoint
    model = StructuredLatentHH(
        checkpoint["latent_size"],
        checkpoint["normalizers"],
        dt_ms=checkpoint["dt_ms"],
        encoder_size=checkpoint.get("encoder_size", 64),
        branch_count=checkpoint.get("branch_count", 3),
        kinetics_width=checkpoint.get("kinetics_width"),
        dynamics_mode=checkpoint.get("dynamics_mode", "relaxation"),
        free_width=checkpoint.get("free_width"),
    ).to(device)
    # Checkpoints trained before the free-dynamics control exist carry no
    # weights for that optional head; anything else must match exactly.
    loaded = model.load_state_dict(checkpoint["model_state"], strict=False)
    unexpected = [key for key in loaded.missing_keys if not key.startswith("free_dynamics.")]
    if unexpected or loaded.unexpected_keys:
        raise RuntimeError(
            f"checkpoint does not match the model: missing={unexpected} "
            f"unexpected={list(loaded.unexpected_keys)}"
        )
    model.eval()
    return model, checkpoint


def _records(root: Path, split: str) -> list[dict]:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line]
    return [record for record in records if record["split"] == split]


def _rollout_record(model, checkpoint, root: Path, record: dict, device, stride: int = 5) -> dict:
    with np.load(root / record["observed_path"]) as archive:
        observed = {name: archive[name] for name in archive.files}
    # Hidden fields are loaded only in this explicitly post-selection module.
    with np.load(root / record["evaluation_path"]) as archive:
        hidden = {name: archive[name] for name in archive.files}
    mode = record["control_mode"]
    if mode == "current_clamp":
        command, response = observed["current_uA_cm2"], observed["voltage_mV"]
    else:
        command, response = observed["voltage_mV"], observed["current_uA_cm2"]
    history_steps = min(checkpoint["history_steps"], len(command) // 3)
    stats = checkpoint["normalizers"][mode]
    command_normalized = (command - stats["command"]["mean"]) / stats["command"]["std"]
    response_normalized = (response - stats["response"]["mean"]) / stats["response"]["std"]
    history = np.stack([command_normalized[:history_steps], response_normalized[:history_steps]], axis=-1)
    future_command = command_normalized[history_steps:]
    with torch.no_grad():
        prediction, latents = model(
            torch.from_numpy(history[None].astype(np.float32)).to(device),
            torch.from_numpy(future_command[None].astype(np.float32)).to(device),
            mode,
            return_latents=True,
        )
    prediction = model.denormalize(prediction, mode, "response").cpu().numpy()[0]
    index = slice(history_steps, None, stride)
    return {
        "mode": mode,
        "target": response[index],
        "prediction": prediction[::stride],
        "activity": observed["activity_score"][index],
        "latents": latents.cpu().numpy()[0, ::stride],
        "gates": np.stack([hidden[name][index] for name in TRUE_GATES], axis=-1),
    }


def _collect(model, checkpoint, root: Path, split: str, device) -> list[dict]:
    return [_rollout_record(model, checkpoint, root, record, device) for record in _records(root, split)]


def _mode_normalizers(checkpoint: dict, mode: str) -> ModeNormalizers:
    values = checkpoint["normalizers"][mode]
    return ModeNormalizers(Standardizer(**values["command"]), Standardizer(**values["response"]))


def _collect_windows(
    model,
    checkpoint: dict,
    root: Path,
    split: str,
    device: torch.device,
    windows_per_mode: int,
    seed: int,
    stride: int = 5,
    strict_sampling: bool = True,
    response_noise_fraction: float = 0.0,
    history_observation_stride: int = 1,
) -> list[dict]:
    """Collect event-balanced rollouts and evaluation-only true gates."""
    if history_observation_stride < 1:
        raise ValueError("history_observation_stride must be positive")
    dataset = ObservedDataset(root, split)
    sampler = BalancedWindowSampler(dataset, checkpoint["history_steps"], checkpoint["prediction_steps"])
    records = {record["trajectory_id"]: record for record in dataset.records}
    hidden_cache: dict[str, dict[str, np.ndarray]] = {}
    collected: list[dict] = []
    rng = torch.Generator(device=device).manual_seed(seed)
    for mode_index, mode in enumerate(model.MODES):
        refs = sampler.sample_mode_epoch(
            mode, windows_per_mode, seed + mode_index, strict=strict_sampling
        )
        normalizers = _mode_normalizers(checkpoint, mode)
        loader = DataLoader(
            WindowTorchDataset(sampler, refs, normalizers), batch_size=128, shuffle=False
        )
        offset = 0
        with torch.no_grad():
            for batch in loader:
                history = batch["history"].to(device)
                command = batch["future_command"].to(device)
                if history_observation_stride > 1:
                    indices = torch.arange(0, history.shape[1], history_observation_stride, device=device)
                    if indices[-1] != history.shape[1] - 1:
                        indices = torch.cat([indices, indices.new_tensor([history.shape[1] - 1])])
                    sparse = history.index_select(1, indices).transpose(1, 2)
                    history = F.interpolate(sparse, size=history.shape[1], mode="linear", align_corners=True).transpose(1, 2)
                if response_noise_fraction > 0.0:
                    history = history.clone()
                    history[:, :, 1] += response_noise_fraction * torch.randn(
                        history[:, :, 1].shape, generator=rng, device=device
                    )
                prediction, latents = model(history, command, mode, return_latents=True)
                prediction = normalizers.response.inverse(prediction.cpu().numpy())
                target = normalizers.response.inverse(batch["target"].numpy())
                command_physical = normalizers.command.inverse(batch["future_command"].numpy())
                history_physical = history.cpu().numpy()
                history_command = normalizers.command.inverse(history_physical[..., 0])
                history_response = normalizers.response.inverse(history_physical[..., 1])
                activity = batch["activity"].numpy()
                batch_refs = refs[offset : offset + len(target)]
                offset += len(target)
                gates = []
                for ref in batch_refs:
                    trajectory_id = ref.trajectory_id
                    if trajectory_id not in hidden_cache:
                        with np.load(root / records[trajectory_id]["evaluation_path"]) as archive:
                            hidden_cache[trajectory_id] = {
                                name: archive[name].copy() for name in TRUE_GATES
                            }
                    hidden = hidden_cache[trajectory_id]
                    window = slice(ref.center, ref.center + checkpoint["prediction_steps"])
                    gates.append(np.stack([hidden[name][window] for name in TRUE_GATES], axis=-1))
                gates_array = np.stack(gates)
                time_slice = slice(None, None, stride)
                collected.append({
                    "mode": mode,
                    "target": target[:, time_slice],
                    "prediction": prediction[:, time_slice],
                    "activity": activity[:, time_slice],
                    "voltage": (
                        target[:, time_slice]
                        if mode == "current_clamp"
                        else command_physical[:, time_slice]
                    ).reshape(-1),
                    "current": (
                        command_physical[:, time_slice]
                        if mode == "current_clamp"
                        else target[:, time_slice]
                    ).reshape(-1),
                    "voltage_slope": np.diff(
                        target[:, time_slice] if mode == "current_clamp" else command_physical[:, time_slice],
                        axis=1,
                        prepend=(target[:, time_slice] if mode == "current_clamp" else command_physical[:, time_slice])[:, :1],
                    ).reshape(-1) / (checkpoint["dt_ms"] * stride),
                    "current_slope": np.diff(
                        command_physical[:, time_slice] if mode == "current_clamp" else target[:, time_slice],
                        axis=1,
                        prepend=(command_physical[:, time_slice] if mode == "current_clamp" else target[:, time_slice])[:, :1],
                    ).reshape(-1) / (checkpoint["dt_ms"] * stride),
                    "voltage_history": _lag_features(
                        target[:, time_slice] if mode == "current_clamp" else command_physical[:, time_slice]
                    ),
                    "current_history": _lag_features(
                        command_physical[:, time_slice] if mode == "current_clamp" else target[:, time_slice]
                    ),
                    "encoder_history_summary": np.repeat(
                        np.column_stack([
                            _history_summary(history_command if mode == "current_clamp" else history_response),
                            _history_summary(history_response if mode == "current_clamp" else history_command),
                        ]),
                        len(range(0, checkpoint["prediction_steps"], stride)),
                        axis=0,
                    ),
                    "encoder_history_grid": np.repeat(
                        np.column_stack([
                            _history_grid(history_command if mode == "current_clamp" else history_response),
                            _history_grid(history_response if mode == "current_clamp" else history_command),
                        ]),
                        len(range(0, checkpoint["prediction_steps"], stride)),
                        axis=0,
                    ),
                    "model_voltage": (
                        prediction[:, time_slice]
                        if mode == "current_clamp"
                        else command_physical[:, time_slice]
                    ).reshape(-1),
                    "trajectory_ids": np.repeat(
                        np.asarray([ref.trajectory_id for ref in batch_refs]),
                        len(range(0, checkpoint["prediction_steps"], stride)),
                    ),
                    "latents": latents.cpu().numpy()[:, time_slice].reshape(-1, model.latent_size),
                    "gates": gates_array[:, time_slice].reshape(-1, len(TRUE_GATES)),
                })
    return collected


def _correlation_matrix(latents: np.ndarray, gates: np.ndarray, method: str) -> np.ndarray:
    result = np.empty((latents.shape[1], gates.shape[1]))
    for i in range(latents.shape[1]):
        for j in range(gates.shape[1]):
            if method == "pearson":
                result[i, j] = np.corrcoef(latents[:, i], gates[:, j])[0, 1]
            else:
                result[i, j] = spearmanr(latents[:, i], gates[:, j]).statistic
    return np.nan_to_num(result)


def blind_evaluate(
    checkpoint_path: str | Path,
    data_root: str | Path,
    mapping_split: str = "validation",
    evaluation_split: str = "test",
    windows_per_mode: int = 300,
    seed: int = 2026,
    device_name: str = "cuda",
    evaluation_data_root: str | Path | None = None,
    ridge_alpha: float = 1e-3,
    ridge_alpha_grid: tuple[float, ...] | None = None,
    ridge_cv_folds: int = 1,
    evaluation_response_noise_fraction: float = 0.0,
    evaluation_history_observation_stride: int = 1,
) -> dict:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    root = Path(data_root)
    evaluation_root = Path(evaluation_data_root) if evaluation_data_root else root
    mapping = _collect_windows(model, checkpoint, root, mapping_split, device, windows_per_mode, seed)
    evaluation = _collect_windows(
        model,
        checkpoint,
        evaluation_root,
        evaluation_split,
        device,
        windows_per_mode,
        seed + 10000,
        strict_sampling=evaluation_root == root,
        response_noise_fraction=evaluation_response_noise_fraction,
        history_observation_stride=evaluation_history_observation_stride,
    )
    mapping_z = np.concatenate([item["latents"] for item in mapping])
    mapping_gates = np.concatenate([item["gates"] for item in mapping])
    mapping_trajectory_ids = np.concatenate([item["trajectory_ids"] for item in mapping])
    evaluation_z = np.concatenate([item["latents"] for item in evaluation])
    evaluation_gates = np.concatenate([item["gates"] for item in evaluation])
    design = np.column_stack([mapping_z, np.ones(len(mapping_z))])
    coefficients = np.linalg.lstsq(design, mapping_gates, rcond=None)[0]
    predicted_gates = np.column_stack([evaluation_z, np.ones(len(evaluation_z))]) @ coefficients
    alpha_selection = None
    if ridge_alpha_grid and ridge_cv_folds > 1:
        unique_ids = np.unique(mapping_trajectory_ids)
        rng = np.random.default_rng(seed + 20000)
        folds = np.array_split(rng.permutation(unique_ids), ridge_cv_folds)
        fold_scores = {str(candidate): [] for candidate in ridge_alpha_grid}
        for calibration_ids in folds:
            calibration_mask = np.isin(mapping_trajectory_ids, calibration_ids)
            fit_mask = ~calibration_mask
            for candidate in ridge_alpha_grid:
                candidate_readout = make_pipeline(
                    StandardScaler(),
                    PolynomialFeatures(degree=3, include_bias=False),
                    Ridge(alpha=candidate),
                )
                candidate_readout.fit(mapping_z[fit_mask], mapping_gates[fit_mask])
                candidate_prediction = candidate_readout.predict(mapping_z[calibration_mask])
                fold_scores[str(candidate)].append(float(np.mean([
                    r2_score(mapping_gates[calibration_mask, index], candidate_prediction[:, index])
                    for index in range(len(TRUE_GATES))
                ])))
        candidate_scores = {key: float(np.mean(values)) for key, values in fold_scores.items()}
        candidate_se = {
            key: float(np.std(values, ddof=1) / np.sqrt(len(values))) for key, values in fold_scores.items()
        }
        best_alpha = max(ridge_alpha_grid, key=lambda value: candidate_scores[str(value)])
        threshold = candidate_scores[str(best_alpha)] - candidate_se[str(best_alpha)]
        ridge_alpha = max(
            value for value in ridge_alpha_grid if candidate_scores[str(value)] >= threshold
        )
        alpha_selection = {
            "method": "trajectory_grouped_cv_one_standard_error",
            "candidate_scores": candidate_scores,
            "candidate_standard_errors": candidate_se,
            "best_mean_alpha": best_alpha,
            "one_se_threshold": threshold,
            "selected_alpha": ridge_alpha,
            "folds": ridge_cv_folds,
            "trajectory_count": int(len(unique_ids)),
        }
    elif ridge_alpha_grid:
        unique_ids = np.unique(mapping_trajectory_ids)
        rng = np.random.default_rng(seed + 20000)
        shuffled_ids = rng.permutation(unique_ids)
        calibration_ids = shuffled_ids[: max(1, len(shuffled_ids) // 5)]
        calibration_mask = np.isin(mapping_trajectory_ids, calibration_ids)
        fit_mask = ~calibration_mask
        candidate_scores = {}
        for candidate in ridge_alpha_grid:
            candidate_readout = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=3, include_bias=False),
                Ridge(alpha=candidate),
            )
            candidate_readout.fit(mapping_z[fit_mask], mapping_gates[fit_mask])
            candidate_prediction = candidate_readout.predict(mapping_z[calibration_mask])
            candidate_scores[str(candidate)] = float(np.mean([
                r2_score(mapping_gates[calibration_mask, index], candidate_prediction[:, index])
                for index in range(len(TRUE_GATES))
            ]))
        ridge_alpha = max(ridge_alpha_grid, key=lambda value: candidate_scores[str(value)])
        alpha_selection = {
            "method": "trajectory_disjoint_id_holdout",
            "candidate_scores": candidate_scores,
            "selected_alpha": ridge_alpha,
            "fit_trajectory_count": int(len(unique_ids) - len(calibration_ids)),
            "calibration_trajectory_count": int(len(calibration_ids)),
        }
    nonlinear_readout = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=ridge_alpha),
    )
    nonlinear_readout.fit(mapping_z, mapping_gates)
    nonlinear_predicted_gates = nonlinear_readout.predict(evaluation_z)
    reverse_readout = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=ridge_alpha),
    )
    reverse_readout.fit(mapping_gates, mapping_z)
    predicted_latents = reverse_readout.predict(evaluation_gates)
    cycled_latents = reverse_readout.predict(nonlinear_predicted_gates)
    pearson = _correlation_matrix(evaluation_z, evaluation_gates, "pearson")
    spearman = _correlation_matrix(evaluation_z, evaluation_gates, "spearman")
    rows, columns = linear_sum_assignment(-np.abs(spearman))
    prediction_by_mode = {}
    for mode in model.MODES:
        items = [item for item in evaluation if item["mode"] == mode]
        target = np.concatenate([item["target"] for item in items])
        prediction = np.concatenate([item["prediction"] for item in items])
        activity = np.concatenate([item["activity"] for item in items])
        prediction_by_mode[mode] = {
            **regression_metrics(target, prediction, activity),
            **waveform_metrics(target, prediction, mode, checkpoint["dt_ms"] * 5),
        }
    return {
        "checkpoint": str(checkpoint_path),
        "mapping_data_root": str(root),
        "evaluation_data_root": str(evaluation_root),
        "mapping_split": mapping_split,
        "evaluation_split": evaluation_split,
        "windows_per_mode": windows_per_mode,
        "seed": seed,
        "ridge_alpha": ridge_alpha,
        "evaluation_response_noise_fraction": float(evaluation_response_noise_fraction),
        "evaluation_history_observation_stride": int(evaluation_history_observation_stride),
        "ridge_alpha_selection": alpha_selection,
        "latent_size": model.latent_size,
        "prediction_by_mode": prediction_by_mode,
        "pearson": pearson.tolist(),
        "spearman": spearman.tolist(),
        "best_monotonic_assignment": [
            {"latent": int(row), "gate": TRUE_GATES[int(column)], "spearman": float(spearman[row, column])}
            for row, column in zip(rows, columns)
        ],
        "linear_mapping_test_r2": {
            gate: float(r2_score(evaluation_gates[:, index], predicted_gates[:, index]))
            for index, gate in enumerate(TRUE_GATES)
        },
        "cubic_mapping_test_r2": {
            gate: float(r2_score(evaluation_gates[:, index], nonlinear_predicted_gates[:, index]))
            for index, gate in enumerate(TRUE_GATES)
        },
        "reverse_cubic_mapping_test_r2": {
            f"z{index}": float(r2_score(evaluation_z[:, index], predicted_latents[:, index]))
            for index in range(model.latent_size)
        },
        "cubic_cycle_test_r2": {
            f"z{index}": float(r2_score(evaluation_z[:, index], cycled_latents[:, index]))
            for index in range(model.latent_size)
        },
    }
