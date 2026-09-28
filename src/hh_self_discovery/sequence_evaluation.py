"""Evaluate a saved black-box baseline without refitting normalizers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .sampling import BalancedWindowSampler, ObservedDataset
from .sequence_models import AutoregressiveSequenceModel
from .training import ModeNormalizers, Standardizer, WindowTorchDataset, evaluate_sequence_model


def evaluate_sequence_checkpoint(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str = "test",
    windows: int = 500,
    prediction_steps: int | None = None,
    seed: int = 1234,
    device_name: str = "cuda",
) -> dict:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AutoregressiveSequenceModel(
        checkpoint["cell_type"], checkpoint["hidden_size"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    values = checkpoint["normalizers"]
    normalizers = ModeNormalizers(
        Standardizer(**values["command"]), Standardizer(**values["response"])
    )
    rollout_steps = prediction_steps or checkpoint["prediction_steps"]
    dataset = ObservedDataset(data_root, split)
    sampler = BalancedWindowSampler(dataset, checkpoint["history_steps"], rollout_steps)
    refs = sampler.sample_mode_epoch(
        checkpoint["control_mode"], windows, seed, strict=False
    )
    loader = DataLoader(
        WindowTorchDataset(sampler, refs, normalizers), batch_size=128, shuffle=False
    )
    return {
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "split": split,
        "control_mode": checkpoint["control_mode"],
        "cell_type": checkpoint["cell_type"],
        "windows": windows,
        "prediction_steps": rollout_steps,
        "free_roll": evaluate_sequence_model(
            model, loader, normalizers, device,
            control_mode=checkpoint["control_mode"]
        ),
        "teacher_forced": evaluate_sequence_model(
            model, loader, normalizers, device,
            teacher_forcing=True, control_mode=checkpoint["control_mode"]
        ),
    }
