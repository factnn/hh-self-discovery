"""Locked observable evaluation for predictive latent baseline checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baseline_models import MODES, build_baseline_model
from .metrics import regression_metrics, waveform_metrics
from .sampling import BalancedWindowSampler, ObservedDataset
from .training import ModeNormalizers, Standardizer, WindowTorchDataset


def _mode_normalizers(checkpoint: dict, mode: str) -> ModeNormalizers:
    values = checkpoint["normalizers"][mode]
    return ModeNormalizers(
        Standardizer(**values["command"]),
        Standardizer(**values["response"]),
    )


def load_baseline_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = build_baseline_model(
        config["model_type"],
        config["latent_size"],
        encoder_width=config["encoder_width"],
        dynamics_width=config["dynamics_width"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def evaluate_predictive_baseline(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str = "test",
    windows_per_mode: int = 500,
    prediction_steps: int | None = None,
    seed: int = 1234,
    device_name: str = "cuda",
) -> dict:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_baseline_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    rollout_steps = prediction_steps or config["prediction_steps"]
    dataset = ObservedDataset(data_root, split)
    sampler = BalancedWindowSampler(
        dataset, config["history_steps"], rollout_steps
    )
    by_mode = {}
    latent_std = {}
    for mode in MODES:
        normalizers = _mode_normalizers(checkpoint, mode)
        references = sampler.sample_mode_epoch(
            mode, windows_per_mode, seed, strict=False
        )
        loader = DataLoader(
            WindowTorchDataset(sampler, references, normalizers),
            batch_size=128,
            shuffle=False,
        )
        targets, predictions, activities, latents = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                history = batch["history"].to(device)
                command = batch["future_command"].to(device)
                prediction, latent = model(
                    history, command, mode, return_latents=True
                )
                predictions.append(
                    normalizers.response.inverse(prediction.cpu().numpy())
                )
                targets.append(
                    normalizers.response.inverse(batch["target"].numpy())
                )
                activities.append(batch["activity"].numpy())
                latents.append(latent.cpu().numpy())
        target = np.concatenate(targets)
        prediction = np.concatenate(predictions)
        activity = np.concatenate(activities)
        latent = np.concatenate(latents)
        by_mode[mode] = {
            **regression_metrics(target, prediction, activity),
            **waveform_metrics(target, prediction, mode, 0.02),
            "normalized_rmse": float(
                np.sqrt(np.mean(np.square(target - prediction)))
                / normalizers.response.std
            ),
        }
        latent_std[mode] = np.std(latent.reshape(-1, latent.shape[-1]), axis=0).tolist()
    return {
        "checkpoint": str(checkpoint_path),
        "model_type": config["model_type"],
        "latent_size": config["latent_size"],
        "split": split,
        "windows_per_mode": windows_per_mode,
        "prediction_steps": rollout_steps,
        "seed": seed,
        "modes": by_mode,
        "mean_normalized_rmse": float(
            np.mean([by_mode[mode]["normalized_rmse"] for mode in MODES])
        ),
        "latent_std": latent_std,
    }
