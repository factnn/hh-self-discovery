"""Estimate global intrinsic dimension of learned latent states using observed I/V only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hh_self_discovery.blind_evaluation import load_latent_checkpoint
from hh_self_discovery.sampling import BalancedWindowSampler, ObservedDataset
from hh_self_discovery.training import ModeNormalizers, Standardizer, WindowTorchDataset


def mode_normalizers(checkpoint: dict, mode: str) -> ModeNormalizers:
    values = checkpoint["normalizers"][mode]
    return ModeNormalizers(
        Standardizer(**values["command"]),
        Standardizer(**values["response"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--windows-per-mode", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=9130)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_latent_checkpoint(args.checkpoint, device)
    dataset = ObservedDataset(args.data, args.split)
    sampler = BalancedWindowSampler(
        dataset, checkpoint["history_steps"], checkpoint["prediction_steps"]
    )
    states = []
    for offset, mode in enumerate(model.MODES):
        refs = sampler.sample_mode_epoch(
            mode, args.windows_per_mode, args.seed + offset, strict=False
        )
        loader = DataLoader(
            WindowTorchDataset(sampler, refs, mode_normalizers(checkpoint, mode)),
            batch_size=256,
            shuffle=False,
        )
        with torch.no_grad():
            for batch in loader:
                states.append(model.encode(batch["history"].to(device), mode).cpu().numpy())

    latent = np.concatenate(states, axis=0).astype(np.float64)
    std = latent.std(axis=0, ddof=1)
    standardized = (latent - latent.mean(axis=0)) / np.maximum(std, 1e-12)
    eigenvalues = np.linalg.eigvalsh(np.cov(standardized, rowvar=False))[::-1]
    fractions = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(fractions)
    result = {
        "checkpoint": args.checkpoint,
        "samples": int(latent.shape[0]),
        "coordinate_std": std.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "explained_fraction": fractions.tolist(),
        "participation_ratio": float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()),
        "entropy_effective_rank": float(np.exp(-(fractions * np.log(np.maximum(fractions, 1e-15))).sum())),
        "pca_dimension_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "pca_dimension_99": int(np.searchsorted(cumulative, 0.99) + 1),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
