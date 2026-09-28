"""Structured latent dynamics with no direct command-to-response bypass."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class StructuredLatentHH(nn.Module):
    """Discover bounded gates and additive conductance-like current branches.

    The two clamp modes use separate history encoders, but share kinetics and
    the current law. All inputs and outputs are normalized at the boundary;
    integration and current evaluation happen in physical HH units.
    """

    MODES = ("current_clamp", "voltage_clamp")

    def __init__(
        self,
        latent_size: int,
        normalizers: dict[str, dict[str, float]],
        dt_ms: float = 0.02,
        encoder_size: int = 64,
        branch_count: int = 3,
        kinetics_width: int | None = None,
        dynamics_mode: str = "relaxation",
        free_width: int | None = None,
    ) -> None:
        super().__init__()
        if latent_size < 1:
            raise ValueError("latent_size must be positive")
        self.latent_size = latent_size
        self.dt_ms = float(dt_ms)
        self.branch_count = branch_count
        if dynamics_mode not in ("relaxation", "free"):
            raise ValueError(f"unknown dynamics_mode: {dynamics_mode}")
        self.dynamics_mode = dynamics_mode
        self.encoders = nn.ModuleDict(
            {
                mode: nn.GRU(2, encoder_size, batch_first=True)
                for mode in self.MODES
            }
        )
        self.initial_state = nn.ModuleDict(
            {mode: nn.Sequential(nn.Linear(encoder_size, latent_size), nn.Sigmoid()) for mode in self.MODES}
        )
        kinetics_width = max(16, 4 * latent_size) if kinetics_width is None else kinetics_width
        if kinetics_width < 1:
            raise ValueError("kinetics_width must be positive")
        self.kinetics_width = kinetics_width
        self.kinetics = nn.Sequential(
            nn.Linear(1, kinetics_width),
            nn.Tanh(),
            nn.Linear(kinetics_width, 2 * latent_size),
        )
        # Same-capacity alternative to the relaxation law: a free vector field
        # dz/dt = f(z, V) with a comparable parameter count and the same bounded
        # coordinates. Used to test whether results depend on the relaxation
        # parameterisation rather than on capacity or data.
        self.free_width = kinetics_width if free_width is None else free_width
        self.free_dynamics = nn.Sequential(
            nn.Linear(latent_size + 1, self.free_width),
            nn.Tanh(),
            nn.Linear(self.free_width, latent_size),
        )
        self.branch_log_g = nn.Parameter(torch.full((branch_count,), math.log(math.expm1(30.0))))
        reversal_init = torch.linspace(-80.0, 50.0, branch_count)
        reversal_fraction = ((reversal_init + 120.0) / 220.0).clamp(0.01, 0.99)
        self.branch_reversal_logits = nn.Parameter(torch.logit(reversal_fraction))
        exponent_init = torch.full((branch_count, latent_size), math.log(math.expm1(0.02)))
        for branch in range(branch_count):
            exponent_init[branch, branch % latent_size] = math.log(math.expm1(1.0))
        self.branch_exponent_raw = nn.Parameter(exponent_init)
        self.leak_log_g = nn.Parameter(torch.tensor(math.log(math.expm1(0.3))))
        leak_fraction = torch.tensor((-54.4 + 120.0) / 220.0)
        self.leak_reversal_logit = nn.Parameter(torch.logit(leak_fraction))
        self._register_normalizers(normalizers)
        self._initialize_kinetics()
        last_free = self.free_dynamics[-1]
        nn.init.normal_(last_free.weight, std=0.02)
        nn.init.zeros_(last_free.bias)

    def _register_normalizers(self, normalizers: dict[str, dict[str, float]]) -> None:
        for mode in self.MODES:
            if mode not in normalizers:
                raise ValueError(f"missing normalizers for {mode}")
            for signal in ("command", "response"):
                for statistic in ("mean", "std"):
                    value = float(normalizers[mode][signal][statistic])
                    self.register_buffer(f"_{mode}_{signal}_{statistic}", torch.tensor(value))

    def _initialize_kinetics(self) -> None:
        last = self.kinetics[-1]
        nn.init.normal_(last.weight, std=0.02)
        nn.init.zeros_(last.bias)
        desired_tau = torch.logspace(-0.7, 1.0, self.latent_size).clamp(0.06, 20.0)
        tau_fraction = ((desired_tau - 0.05) / 49.95).clamp(1e-4, 1.0 - 1e-4)
        with torch.no_grad():
            last.bias[self.latent_size :] = torch.logit(tau_fraction)

    def _stat(self, mode: str, signal: str, statistic: str) -> torch.Tensor:
        return getattr(self, f"_{mode}_{signal}_{statistic}")

    def denormalize(self, values: torch.Tensor, mode: str, signal: str) -> torch.Tensor:
        return values * self._stat(mode, signal, "std") + self._stat(mode, signal, "mean")

    def normalize(self, values: torch.Tensor, mode: str, signal: str) -> torch.Tensor:
        return (values - self._stat(mode, signal, "mean")) / self._stat(mode, signal, "std")

    def encode(self, history: torch.Tensor, mode: str) -> torch.Tensor:
        if mode not in self.MODES:
            raise ValueError(f"unknown control mode: {mode}")
        _, hidden = self.encoders[mode](history)
        return self.initial_state[mode](hidden[-1])

    def gate_kinetics(self, voltage_mV: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        voltage_scaled = ((voltage_mV.clamp(-150.0, 100.0) + 50.0) / 100.0).unsqueeze(-1)
        raw = self.kinetics(voltage_scaled)
        steady = torch.sigmoid(raw[..., : self.latent_size])
        tau_ms = 0.05 + 49.95 * torch.sigmoid(raw[..., self.latent_size :])
        return steady, tau_ms

    def advance_gates(self, gates: torch.Tensor, voltage_mV: torch.Tensor) -> torch.Tensor:
        if self.dynamics_mode == "free":
            scaled = ((voltage_mV.clamp(-150.0, 100.0) + 50.0) / 100.0).unsqueeze(-1)
            rate = self.free_dynamics(torch.cat((gates, scaled), dim=-1))
            return (gates + self.dt_ms * rate).clamp(1e-4, 1.0 - 1e-4)
        steady, tau_ms = self.gate_kinetics(voltage_mV)
        decay = torch.exp(-self.dt_ms / tau_ms)
        return steady + (gates - steady) * decay

    def current_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "conductance": F.softplus(self.branch_log_g),
            "reversal_mV": -120.0 + 220.0 * torch.sigmoid(self.branch_reversal_logits),
            "exponents": F.softplus(self.branch_exponent_raw),
            "leak_conductance": F.softplus(self.leak_log_g),
            "leak_reversal_mV": -120.0 + 220.0 * torch.sigmoid(self.leak_reversal_logit),
        }

    def ionic_current(self, voltage_mV: torch.Tensor, gates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.current_parameters()
        log_gates = torch.log(gates.clamp(1e-5, 1.0)).unsqueeze(1)
        activation = torch.exp((params["exponents"].unsqueeze(0) * log_gates).sum(dim=-1))
        branches = params["conductance"].unsqueeze(0) * activation * (
            voltage_mV.unsqueeze(-1) - params["reversal_mV"].unsqueeze(0)
        )
        leak = params["leak_conductance"] * (voltage_mV - params["leak_reversal_mV"])
        return branches.sum(dim=-1) + leak, branches

    def complexity_penalty(self) -> torch.Tensor:
        params = self.current_parameters()
        state_usage = torch.sqrt(params["exponents"].square().sum(dim=0) + 1e-8).sum()
        branch_usage = params["conductance"].sum()
        return 1e-3 * state_usage + 1e-6 * branch_usage

    def rollout_from_state(
        self,
        gates: torch.Tensor,
        previous_response: torch.Tensor,
        future_command: torch.Tensor,
        mode: str,
        return_latents: bool = False,
        ablate_index: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Roll out from an explicit latent state for observability analysis.

        The future command is normalized at the model boundary, while the
        previous response is in physical units. No latent labels are used.
        """
        if mode not in self.MODES:
            raise ValueError(f"unknown control mode: {mode}")
        command = self.denormalize(future_command, mode, "command")
        predictions = []
        latent_trace = []
        for step in range(command.shape[1]):
            if mode == "current_clamp":
                voltage = previous_response.clamp(-150.0, 100.0)
                gates = self.advance_gates(gates, voltage)
                if ablate_index is not None:
                    gates = gates.clone()
                    gates[:, ablate_index] = 0.5
                ionic, _ = self.ionic_current(voltage, gates)
                derivative = (command[:, step] - ionic).clamp(-1000.0, 1000.0)
                response = (voltage + self.dt_ms * derivative).clamp(-150.0, 100.0)
            else:
                voltage = command[:, step]
                gates = self.advance_gates(gates, voltage)
                if ablate_index is not None:
                    gates = gates.clone()
                    gates[:, ablate_index] = 0.5
                response, _ = self.ionic_current(voltage, gates)
            predictions.append(self.normalize(response, mode, "response"))
            latent_trace.append(gates)
            previous_response = response
        prediction = torch.stack(predictions, dim=1)
        if return_latents:
            return prediction, torch.stack(latent_trace, dim=1)
        return prediction

    def forward(
        self,
        history: torch.Tensor,
        future_command: torch.Tensor,
        mode: str,
        return_latents: bool = False,
        ablate_index: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        gates = self.encode(history, mode)
        previous_response = self.denormalize(history[:, -1, 1], mode, "response")
        return self.rollout_from_state(
            gates,
            previous_response,
            future_command,
            mode,
            return_latents=return_latents,
            ablate_index=ablate_index,
        )
