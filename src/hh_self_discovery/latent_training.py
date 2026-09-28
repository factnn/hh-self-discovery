"""Joint current/voltage-clamp training for structured latent models."""

from __future__ import annotations

import json
import hashlib
import math
import platform
import random
from dataclasses import asdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .latent_model import StructuredLatentHH
from .sampling import BalancedWindowSampler, ObservedDataset
from .events import EventClass
from .training import ModeNormalizers, WindowTorchDataset, _loss, fit_mode_normalizers



def curriculum_rollout_steps(
    prediction_steps: int,
    epoch: int,
    epochs: int,
    curriculum_fraction: float,
) -> int:
    """Reach the full horizon after a controlled fraction of training."""
    if not 0.0 < curriculum_fraction <= 1.0:
        raise ValueError("curriculum_fraction must be in (0, 1]")
    ramp_epochs = max(1, int(math.ceil(epochs * curriculum_fraction)))
    progress = min(1.0, (epoch + 1) / ramp_epochs)
    return min(prediction_steps, max(25, int(round(prediction_steps * progress))))


def multi_horizon_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    """Keep short-horizon accuracy visible while optimizing long rollouts."""
    steps = prediction.shape[1]
    horizons = sorted({max(1, steps // 4), max(1, steps // 2), steps})
    return torch.stack([
        _loss(prediction[:, :horizon], target[:, :horizon], activity[:, :horizon])
        for horizon in horizons
    ]).mean()


def spike_waveform_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    temperature: float = 0.15,
) -> torch.Tensor:
    """Differentiable spike-shape loss using voltage observations only."""
    predicted_active = torch.sigmoid((prediction - threshold) / temperature)
    target_active = torch.sigmoid((target - threshold) / temperature)
    threshold_loss = (predicted_active - target_active).square().mean()
    peak_loss = (prediction.amax(dim=1) - target.amax(dim=1)).square().mean()
    if prediction.shape[1] > 1:
        predicted_delta = prediction[:, 1:] - prediction[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        rapid_weight = 1.0 + 8.0 * target_delta.abs() / (
            target_delta.abs().mean(dim=1, keepdim=True) + 1e-6
        )
        rapid_loss = (
            rapid_weight * (predicted_delta - target_delta).square()
        ).sum() / rapid_weight.sum()
    else:
        rapid_loss = prediction.new_zeros(())
    return threshold_loss + 0.25 * peak_loss + 0.5 * rapid_loss


def conditional_spike_waveform_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    temperature: float = 0.15,
) -> torch.Tensor:
    """Separate observable spiking windows from non-spiking windows."""
    target_has_spike = target.amax(dim=1) >= threshold
    terms = []
    if target_has_spike.any():
        terms.append(
            spike_waveform_loss(
                prediction[target_has_spike],
                target[target_has_spike],
                threshold,
                temperature,
            )
        )
    if (~target_has_spike).any():
        false_positive = torch.sigmoid(
            (prediction[~target_has_spike] - threshold) / temperature
        )
        terms.append(false_positive.square().mean())
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())


def post_command_transient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    command: torch.Tensor,
    transient_steps: int,
    jump_threshold: float = 0.05,
) -> torch.Tensor:
    """Concentrate response error immediately after observable command jumps."""
    if prediction.shape[1] < 2 or transient_steps < 1:
        return prediction.new_zeros(())
    jumps = prediction.new_zeros(prediction.shape, dtype=torch.bool)
    jumps[:, 1:] = (command[:, 1:] - command[:, :-1]).abs() >= jump_threshold
    mask = torch.zeros_like(jumps)
    for offset in range(transient_steps):
        if offset == 0:
            mask |= jumps
        else:
            mask[:, offset:] |= jumps[:, :-offset]
    if not mask.any():
        return prediction.new_zeros(())
    return (prediction[mask] - target[mask]).square().mean()


def _reproducibility_metadata(device: torch.device) -> dict:
    source_root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("latent_model.py", "latent_training.py", "sampling.py", "training.py"):
        digest.update((source_root / name).read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
    }


def _normalizer_dict(normalizers: dict[str, ModeNormalizers]) -> dict:
    return {
        mode: {
            "command": asdict(values.command),
            "response": asdict(values.response),
        }
        for mode, values in normalizers.items()
    }


def _evaluate(model, loaders, normalizers, device) -> dict:
    result = {}
    for mode, loader in loaders.items():
        # StructuredLatentHH has the same normalized boundary as sequence models.
        model.eval()
        grouped_target, grouped_prediction, grouped_activity = [], [], []
        with torch.no_grad():
            for batch in loader:
                history = batch["history"].to(device)
                command = batch["future_command"].to(device)
                prediction = model(history, command, mode).cpu().numpy()
                target = batch["target"].numpy()
                grouped_prediction.append(normalizers[mode].response.inverse(prediction))
                grouped_target.append(normalizers[mode].response.inverse(target))
                grouped_activity.append(batch["activity"].numpy())
        from .metrics import regression_metrics, waveform_metrics

        target = np.concatenate(grouped_target)
        prediction = np.concatenate(grouped_prediction)
        activity = np.concatenate(grouped_activity)
        result[mode] = {
            "windows": len(target),
            **regression_metrics(target, prediction, activity),
            **waveform_metrics(target, prediction, mode, model.dt_ms),
            "normalized_rmse": float(np.sqrt(np.mean(np.square(target - prediction))) / normalizers[mode].response.std),
        }
    return result


CURRENT_EVENT_PROFILES = {
    "default": None,
    "spike_focused": {
        EventClass.STEADY: 0.05,
        EventClass.COMMAND: 0.10,
        EventClass.FAST_RESPONSE: 0.25,
        EventClass.ACTIVE: 0.40,
        EventClass.RECOVERY: 0.20,
    },
}


def train_structured_latent(
    data_root: str,
    output_dir: str,
    latent_size: int,
    encoder_size: int = 64,
    branch_count: int = 3,
    kinetics_width: int | None = None,
    dynamics_mode: str = "relaxation",
    free_width: int | None = None,
    epochs: int = 10,
    train_windows_per_mode: int = 5000,
    validation_windows_per_mode: int = 1000,
    history_steps: int = 500,
    prediction_steps: int = 250,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    complexity_scale: float = 1.0,
    weight_decay: float = 1e-6,
    curriculum_fraction: float = 1.0,
    use_multi_horizon_loss: bool = False,
    spike_loss_weight: float = 0.0,
    spike_loss_mode: str = "global",
    current_clamp_loss_weight: float = 1.0,
    voltage_transient_loss_weight: float = 0.0,
    voltage_transient_steps: int = 25,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    current_event_profile: str = "default",
    seed: int = 42,
    device_name: str = "cuda",
) -> dict:
    if encoder_size < 1:
        raise ValueError("encoder_size must be positive")
    if branch_count < 1:
        raise ValueError("branch_count must be positive")
    if kinetics_width is not None and kinetics_width < 1:
        raise ValueError("kinetics_width must be positive")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if current_clamp_loss_weight <= 0.0:
        raise ValueError("current_clamp_loss_weight must be positive")
    if voltage_transient_loss_weight < 0.0 or voltage_transient_steps < 1:
        raise ValueError("voltage transient loss settings must be non-negative with positive steps")
    if spike_loss_weight < 0.0:
        raise ValueError("spike_loss_weight must be non-negative")
    if spike_loss_mode not in {"global", "conditional"}:
        raise ValueError("spike_loss_mode must be global or conditional")
    if current_event_profile not in CURRENT_EVENT_PROFILES:
        raise ValueError(f"unknown current_event_profile: {current_event_profile}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    train_data = ObservedDataset(data_root, "train")
    validation_data = ObservedDataset(data_root, "validation")
    modes = StructuredLatentHH.MODES
    normalizers = {mode: fit_mode_normalizers(train_data, mode) for mode in modes}
    train_samplers = {
        mode: BalancedWindowSampler(
            train_data,
            history_steps,
            prediction_steps,
            fractions=(
                CURRENT_EVENT_PROFILES[current_event_profile]
                if mode == "current_clamp"
                else None
            ),
        )
        for mode in modes
    }
    validation_samplers = {mode: BalancedWindowSampler(validation_data, history_steps, prediction_steps) for mode in modes}
    validation_loaders = {}
    for mode in modes:
        refs = validation_samplers[mode].sample_mode_epoch(mode, validation_windows_per_mode, seed + 10000)
        validation_loaders[mode] = DataLoader(
            WindowTorchDataset(validation_samplers[mode], refs, normalizers[mode]),
            batch_size=batch_size,
            shuffle=False,
        )
    model = StructuredLatentHH(
        latent_size,
        _normalizer_dict(normalizers),
        encoder_size=encoder_size,
        branch_count=branch_count,
        kinetics_width=kinetics_width,
        dynamics_mode=dynamics_mode,
        free_width=free_width,
    ).to(device)
    physical_parameters = [
        model.branch_log_g,
        model.branch_reversal_logits,
        model.branch_exponent_raw,
        model.leak_log_g,
        model.leak_reversal_logit,
    ]
    physical_ids = {id(parameter) for parameter in physical_parameters}
    dynamical_parameters = [parameter for parameter in model.parameters() if id(parameter) not in physical_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": dynamical_parameters, "lr": learning_rate},
            {"params": physical_parameters, "lr": 5.0 * learning_rate},
        ],
        weight_decay=weight_decay,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "latent_size": latent_size,
        "encoder_size": encoder_size,
        "branch_count": branch_count,
        "kinetics_width": model.kinetics_width,
        "dynamics_mode": model.dynamics_mode,
        "free_width": model.free_width,
        "epochs": epochs,
        "train_windows_per_mode": train_windows_per_mode,
        "validation_windows_per_mode": validation_windows_per_mode,
        "history_steps": history_steps,
        "prediction_steps": prediction_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "complexity_scale": complexity_scale,
        "curriculum_fraction": curriculum_fraction,
        "use_multi_horizon_loss": use_multi_horizon_loss,
        "spike_loss_weight": spike_loss_weight,
        "spike_loss_mode": spike_loss_mode,
        "current_clamp_loss_weight": current_clamp_loss_weight,
        "voltage_transient_loss_weight": voltage_transient_loss_weight,
        "voltage_transient_steps": voltage_transient_steps,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "current_event_profile": current_event_profile,
        "current_event_fractions": (
            None
            if CURRENT_EVENT_PROFILES[current_event_profile] is None
            else {
                event.name: fraction
                for event, fraction in CURRENT_EVENT_PROFILES[current_event_profile].items()
            }
        ),
        "weight_decay": weight_decay,
        "device_name": device_name,
    }
    reproducibility = _reproducibility_metadata(device)
    history_records = []
    best_score = float("inf")
    stopping_best_score = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(epochs):
        rollout_steps = curriculum_rollout_steps(
            prediction_steps, epoch, epochs, curriculum_fraction
        )
        loaders = {}
        for mode in modes:
            refs = train_samplers[mode].sample_mode_epoch(mode, train_windows_per_mode, seed + epoch)
            loaders[mode] = DataLoader(
                WindowTorchDataset(train_samplers[mode], refs, normalizers[mode]),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + epoch),
            )
        model.train()
        losses = []
        latent_stds = []
        batches = zip_longest(loaders[modes[0]], loaders[modes[1]])
        for pair in batches:
            for mode, batch in zip(modes, pair):
                if batch is None:
                    continue
                history_tensor = batch["history"].to(device)
                command = batch["future_command"][:, :rollout_steps].to(device)
                target = batch["target"][:, :rollout_steps].to(device)
                activity = batch["activity"][:, :rollout_steps].to(device)
                prediction, latents = model(history_tensor, command, mode, return_latents=True)
                prediction_loss = (
                    multi_horizon_loss(prediction, target, activity)
                    if use_multi_horizon_loss
                    else _loss(prediction, target, activity)
                )
                if mode == "current_clamp" and spike_loss_weight > 0.0:
                    voltage_normalizer = normalizers[mode].response
                    normalized_threshold = (
                        0.0 - voltage_normalizer.mean
                    ) / voltage_normalizer.std
                    spike_loss_fn = (
                        conditional_spike_waveform_loss
                        if spike_loss_mode == "conditional"
                        else spike_waveform_loss
                    )
                    prediction_loss = prediction_loss + spike_loss_weight * spike_loss_fn(
                        prediction, target, normalized_threshold
                    )
                if mode == "voltage_clamp" and voltage_transient_loss_weight > 0.0:
                    prediction_loss = prediction_loss + voltage_transient_loss_weight * post_command_transient_loss(
                        prediction, target, command, voltage_transient_steps
                    )
                mode_weight = (
                    current_clamp_loss_weight if mode == "current_clamp" else 1.0
                )
                loss = mode_weight * prediction_loss + complexity_scale * model.complexity_penalty()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                latent_stds.append(latents.detach().std(dim=(0, 1)).cpu().numpy())
        metrics = _evaluate(model, validation_loaders, normalizers, device)
        score = float(np.mean([metrics[mode]["normalized_rmse"] for mode in modes]))
        parameters = {key: value.detach().cpu().tolist() for key, value in model.current_parameters().items()}
        record = {
            "epoch": epoch + 1,
            "rollout_steps": rollout_steps,
            "train_loss": float(np.mean(losses)),
            "latent_std": np.mean(latent_stds, axis=0).tolist(),
            "validation": metrics,
            "selection_score": score,
            "current_parameters": parameters,
        }
        history_records.append(record)
        print(json.dumps(record, sort_keys=True))
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "latent_size": latent_size,
                    "encoder_size": encoder_size,
                    "branch_count": branch_count,
                    "kinetics_width": model.kinetics_width,
                    "dynamics_mode": model.dynamics_mode,
                    "free_width": model.free_width,
                    "normalizers": _normalizer_dict(normalizers),
                    "history_steps": history_steps,
                    "prediction_steps": prediction_steps,
                    "dt_ms": model.dt_ms,
                    "validation": metrics,
                    "seed": seed,
                    "config": config,
                    "reproducibility": reproducibility,
                },
                output / "best.pt",
            )
        # Curriculum epochs use different training horizons. Start patience only
        # once the requested full rollout is active, while always selecting the
        # checkpoint on the same fixed, full-horizon validation set.
        if early_stopping_patience and rollout_steps == prediction_steps:
            if score < stopping_best_score - early_stopping_min_delta:
                stopping_best_score = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                stopped_early = True
                break
    summary = {
        "latent_size": latent_size,
        "seed": seed,
        "epochs": epochs,
        "epochs_completed": len(history_records),
        "stopped_early": stopped_early,
        "best_selection_score": best_score,
        "config": config,
        "reproducibility": reproducibility,
        "history": history_records,
    }
    (output / "training.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
