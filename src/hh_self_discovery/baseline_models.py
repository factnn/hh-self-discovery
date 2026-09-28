"""Unified predictive latent baselines built from pinned upstream components."""

from __future__ import annotations

import torch
from torch import nn


MODES = ("current_clamp", "voltage_clamp")


class _GRUEncoder(nn.Module):
    def __init__(self, latent_size: int, width: int) -> None:
        super().__init__()
        self.recurrent = nn.GRU(2, width, batch_first=True)
        self.readout = nn.Linear(width, latent_size)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.recurrent(history)
        return torch.tanh(self.readout(hidden[-1]))


class _S4DEncoder(nn.Module):
    """History encoder using the official pinned S4D layer."""

    def __init__(self, latent_size: int, width: int, layers: int = 2) -> None:
        super().__init__()
        try:
            from models.s4.s4d import S4D
        except ImportError as error:
            raise ImportError(
                "S4D requires PYTHONPATH to include external/s4"
            ) from error
        self.input = nn.Linear(2, width)
        self.blocks = nn.ModuleList(
            [S4D(width, d_state=64, dropout=0.0, transposed=True) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.readout = nn.Linear(width, latent_size)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        values = self.input(history)
        for block, norm in zip(self.blocks, self.norms):
            update, _ = block(values.transpose(1, 2))
            values = norm(values + update.transpose(1, 2))
        return torch.tanh(self.readout(values[:, -1]))


class PredictiveBottleneck(nn.Module):
    """Generic controlled latent state-space model with a K-dimensional bottleneck."""

    MODES = MODES

    def __init__(
        self,
        latent_size: int,
        encoder_type: str = "gru",
        encoder_width: int = 128,
        transition_width: int = 128,
        dt_ms: float = 0.02,
    ) -> None:
        super().__init__()
        if encoder_type not in {"gru", "s4d"}:
            raise ValueError("encoder_type must be gru or s4d")
        encoder_cls = _GRUEncoder if encoder_type == "gru" else _S4DEncoder
        self.encoders = nn.ModuleDict(
            {mode: encoder_cls(latent_size, encoder_width) for mode in MODES}
        )
        self.transitions = nn.ModuleDict(
            {
                mode: nn.Sequential(
                    nn.Linear(latent_size + 2, transition_width),
                    nn.Tanh(),
                    nn.Linear(transition_width, latent_size),
                )
                for mode in MODES
            }
        )
        self.outputs = nn.ModuleDict(
            {
                mode: nn.Sequential(
                    nn.Linear(latent_size + 2, transition_width),
                    nn.Tanh(),
                    nn.Linear(transition_width, 1),
                )
                for mode in MODES
            }
        )
        self.latent_size = latent_size
        self.encoder_type = encoder_type
        self.encoder_width = encoder_width
        self.transition_width = transition_width
        self.dt_ms = float(dt_ms)
        self._initialize_rollout()

    def _initialize_rollout(self) -> None:
        for network in list(self.transitions.values()) + list(self.outputs.values()):
            nn.init.normal_(network[-1].weight, std=1e-3)
            nn.init.zeros_(network[-1].bias)

    def encode(self, history: torch.Tensor, mode: str) -> torch.Tensor:
        return self.encoders[mode](history)

    def forward(
        self,
        history: torch.Tensor,
        future_command: torch.Tensor,
        mode: str,
        return_latents: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(history, mode)
        previous = history[:, -1, 1]
        predictions, latents = [], []
        for step in range(future_command.shape[1]):
            command = future_command[:, step]
            context = torch.cat((latent, command[:, None], previous[:, None]), dim=-1)
            latent = latent + 0.1 * torch.tanh(self.transitions[mode](context))
            output_context = torch.cat((latent, command[:, None], previous[:, None]), dim=-1)
            previous = previous + self.outputs[mode](output_context).squeeze(-1)
            predictions.append(previous)
            latents.append(latent)
        prediction = torch.stack(predictions, dim=1)
        if return_latents:
            return prediction, torch.stack(latents, dim=1)
        return prediction


class _ControlledODEFunc(nn.Module):
    def __init__(self, state_size: int, width: int, dt_ms: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size + 2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, state_size),
        )
        self.dt_ms = float(dt_ms)
        self.command: torch.Tensor | None = None
        self.mode_value = 0.0
        nn.init.normal_(self.network[-1].weight, std=1e-3)
        nn.init.zeros_(self.network[-1].bias)

    def set_context(self, command: torch.Tensor, mode: str) -> None:
        self.command = command
        self.mode_value = -1.0 if mode == "current_clamp" else 1.0

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.command is None:
            raise RuntimeError("controlled ODE context is unset")
        index = min(
            self.command.shape[1] - 1,
            max(0, int(torch.floor(time.detach() / self.dt_ms + 1e-6).item())),
        )
        command = self.command[:, index]
        prefix = state.shape[:-1]
        command = command.reshape((1,) * (len(prefix) - 1) + (command.shape[0], 1))
        command = command.expand(*prefix, 1)
        mode = state.new_full((*prefix, 1), self.mode_value)
        return self.network(torch.cat((state, command, mode), dim=-1))


class ControlledLatentODE(nn.Module):
    """Controlled HH adapter retaining the official Latent ODE recognition/solver code."""

    MODES = MODES

    def __init__(
        self,
        latent_size: int,
        encoder_width: int = 128,
        dynamics_width: int = 128,
        dt_ms: float = 0.02,
        method: str = "euler",
    ) -> None:
        super().__init__()
        try:
            from lib.diffeq_solver import DiffeqSolver
            from lib.encoder_decoder import Encoder_z0_RNN
        except ImportError as error:
            raise ImportError(
                "Latent ODE requires PYTHONPATH to include external/latent_ode"
            ) from error
        self.encoders = nn.ModuleDict(
            {
                mode: Encoder_z0_RNN(
                    latent_dim=latent_size,
                    input_dim=4,
                    lstm_output_size=encoder_width,
                    use_delta_t=True,
                )
                for mode in MODES
            }
        )
        self.ode_func = _ControlledODEFunc(latent_size + 1, dynamics_width, dt_ms)
        self.solver = DiffeqSolver(
            input_dim=latent_size + 1,
            ode_func=self.ode_func,
            method=method,
            latents=latent_size + 1,
        )
        self.latent_size = latent_size
        self.encoder_width = encoder_width
        self.dynamics_width = dynamics_width
        self.method = method
        self.dt_ms = float(dt_ms)

    def encode(self, history: torch.Tensor, mode: str) -> torch.Tensor:
        encoder = self.encoders[mode]
        encoder.device = history.device
        time = torch.arange(history.shape[1], device=history.device, dtype=history.dtype) * self.dt_ms
        observed_with_mask = torch.cat((history, torch.ones_like(history)), dim=-1)
        mean, _ = encoder(observed_with_mask, time, run_backwards=True)
        return mean[0]

    def forward(
        self,
        history: torch.Tensor,
        future_command: torch.Tensor,
        mode: str,
        return_latents: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(history, mode)
        first = torch.cat((latent, history[:, -1, 1:2]), dim=-1).unsqueeze(0)
        self.ode_func.set_context(future_command, mode)
        time = torch.arange(
            future_command.shape[1] + 1,
            device=history.device,
            dtype=history.dtype,
        ) * self.dt_ms
        trajectory = self.solver(first, time)[0, :, 1:]
        prediction = trajectory[..., -1]
        latents = trajectory[..., : self.latent_size]
        if return_latents:
            return prediction, latents
        return prediction


def build_baseline_model(
    model_type: str,
    latent_size: int,
    encoder_width: int = 128,
    dynamics_width: int = 128,
    dt_ms: float = 0.02,
) -> nn.Module:
    if model_type in {"gru_bottleneck", "s4d_bottleneck"}:
        return PredictiveBottleneck(
            latent_size=latent_size,
            encoder_type="gru" if model_type.startswith("gru") else "s4d",
            encoder_width=encoder_width,
            transition_width=dynamics_width,
            dt_ms=dt_ms,
        )
    if model_type == "controlled_latent_ode":
        return ControlledLatentODE(
            latent_size=latent_size,
            encoder_width=encoder_width,
            dynamics_width=dynamics_width,
            dt_ms=dt_ms,
        )
    raise ValueError(f"unknown baseline model type: {model_type}")
