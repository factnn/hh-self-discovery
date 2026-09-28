"""Observable-only output-sensitivity metrics for learned latent dynamics."""

from __future__ import annotations

import numpy as np
import torch

from .latent_model import StructuredLatentHH


def output_jacobian(
    model: StructuredLatentHH,
    history: torch.Tensor,
    future_command: torch.Tensor,
    mode: str,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Finite-difference normalized output with respect to encoded state."""
    if history.shape[0] != 1:
        raise ValueError("observability windows must be evaluated one at a time")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    with torch.no_grad():
        state = model.encode(history, mode)
        previous = model.denormalize(history[:, -1, 1], mode, "response")
        latent_size = state.shape[1]
        perturbation = torch.eye(latent_size, device=state.device) * epsilon
        states = torch.cat(
            [(state + perturbation).clamp(0.0, 1.0), (state - perturbation).clamp(0.0, 1.0)]
        )
        commands = future_command.repeat(2 * latent_size, 1)
        previous_values = previous.repeat(2 * latent_size)
        outputs = model.rollout_from_state(states, previous_values, commands, mode)
        plus, minus = outputs[:latent_size], outputs[latent_size:]
        denominators = (states[:latent_size] - states[latent_size:]).diagonal()
        return ((plus - minus) / denominators[:, None]).transpose(0, 1)


def gramian_metrics(jacobian: np.ndarray, relative_tolerance: float = 1e-4) -> dict:
    """Summarize the finite-horizon latent output-sensitivity Gramian J.T @ J."""
    values = np.asarray(jacobian, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("jacobian must have shape [time, latent]")
    gramian = values.T @ values / values.shape[0]
    eigenvalues = np.linalg.eigvalsh(gramian).clip(0.0)
    largest = max(float(eigenvalues[-1]), 1e-15)
    rank = int(np.sum(eigenvalues > relative_tolerance * largest))
    smallest = float(eigenvalues[0])
    return {
        "eigenvalues": eigenvalues.tolist(),
        "effective_rank": rank,
        "minimum_eigenvalue": smallest,
        "log_determinant": float(np.linalg.slogdet(gramian + 1e-12 * np.eye(values.shape[1]))[1]),
        "condition_number": float(largest / max(smallest, 1e-15)),
    }
