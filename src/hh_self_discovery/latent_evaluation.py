"""Observable-only ablation and noise robustness for locked latent models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .blind_evaluation import load_latent_checkpoint
from .metrics import regression_metrics, waveform_metrics
from .sampling import BalancedWindowSampler, ObservedDataset
from .training import ModeNormalizers, Standardizer, WindowTorchDataset


def _mode_normalizers(checkpoint: dict, mode: str) -> ModeNormalizers:
    values = checkpoint["normalizers"][mode]
    return ModeNormalizers(Standardizer(**values["command"]), Standardizer(**values["response"]))


def evaluate_locked_latent(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str = "validation",
    windows_per_mode: int = 500,
    seed: int = 1234,
    response_noise_fraction: float = 0.0,
    prediction_steps: int | None = None,
    include_ablations: bool = True,
    history_observation_stride: int = 1,
    device_name: str = "cuda",
) -> dict:
    if history_observation_stride < 1:
        raise ValueError("history_observation_stride must be positive")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(checkpoint_path, device)
    dataset = ObservedDataset(data_root, split)
    rollout_steps = prediction_steps or checkpoint["prediction_steps"]
    sampler = BalancedWindowSampler(dataset, checkpoint["history_steps"], rollout_steps)
    def evaluate(ablate_index: int | None) -> dict:
        # Make all ablations see exactly the same noisy histories.
        rng = torch.Generator(device=device).manual_seed(seed)
        by_mode = {}
        for mode in model.MODES:
            normalizers = _mode_normalizers(checkpoint, mode)
            refs = sampler.sample_mode_epoch(mode, windows_per_mode, seed, strict=False)
            loader = DataLoader(WindowTorchDataset(sampler, refs, normalizers), batch_size=128, shuffle=False)
            targets, predictions, activities = [], [], []
            with torch.no_grad():
                for batch in loader:
                    history = batch["history"].to(device)
                    command = batch["future_command"].to(device)
                    if history_observation_stride > 1:
                        indices = torch.arange(
                            0, history.shape[1], history_observation_stride, device=device
                        )
                        if indices[-1] != history.shape[1] - 1:
                            indices = torch.cat([indices, indices.new_tensor([history.shape[1] - 1])])
                        sparse = history.index_select(1, indices).transpose(1, 2)
                        history = F.interpolate(
                            sparse, size=history.shape[1], mode="linear", align_corners=True
                        ).transpose(1, 2)
                    if response_noise_fraction > 0.0:
                        history = history.clone()
                        history[:, :, 1] += response_noise_fraction * torch.randn(
                            history[:, :, 1].shape, generator=rng, device=device
                        )
                    prediction = model(history, command, mode, ablate_index=ablate_index).cpu().numpy()
                    target = batch["target"].numpy()
                    predictions.append(normalizers.response.inverse(prediction))
                    targets.append(normalizers.response.inverse(target))
                    activities.append(batch["activity"].numpy())
            target = np.concatenate(targets)
            prediction = np.concatenate(predictions)
            activity = np.concatenate(activities)
            by_mode[mode] = {
                **regression_metrics(target, prediction, activity),
                **waveform_metrics(target, prediction, mode, checkpoint["dt_ms"]),
                "normalized_rmse": float(np.sqrt(np.mean((target - prediction) ** 2)) / normalizers.response.std),
            }
        by_mode["mean_normalized_rmse"] = float(
            np.mean([by_mode[mode]["normalized_rmse"] for mode in model.MODES])
        )
        return by_mode

    baseline = evaluate(None)
    ablations = []
    if include_ablations:
        for index in range(model.latent_size):
            result = evaluate(index)
            result["latent"] = index
            result["relative_degradation"] = (
                result["mean_normalized_rmse"] / baseline["mean_normalized_rmse"] - 1.0
            )
            ablations.append(result)
    return {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "windows_per_mode": windows_per_mode,
        "prediction_steps": rollout_steps,
        "response_noise_fraction": response_noise_fraction,
        "history_observation_stride": history_observation_stride,
        "baseline": baseline,
        "ablations": ablations,
    }
