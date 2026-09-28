"""Autoregressive black-box sequence baselines."""

from __future__ import annotations

import torch
from torch import nn


class AutoregressiveSequenceModel(nn.Module):
    """Encode observed command/response history and freely roll out a response."""

    def __init__(self, cell_type: str = "rnn", hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        if cell_type not in {"mlp", "rnn", "gru"}:
            raise ValueError("cell_type must be 'mlp', 'rnn', or 'gru'")
        if cell_type == "mlp":
            self.encoder = None
            self.decoder_cell = None
            self.output = nn.Sequential(nn.Linear(2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        else:
            recurrent = nn.RNN if cell_type == "rnn" else nn.GRU
            recurrent_cell = nn.RNNCell if cell_type == "rnn" else nn.GRUCell
            self.encoder = recurrent(2, hidden_size, num_layers=num_layers, batch_first=True)
            self.decoder_cell = recurrent_cell(2, hidden_size)
            self.output = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        # Start as the persistence model. Training then learns only corrections,
        # rather than first having to discover the identity rollout.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        history: torch.Tensor,
        future_command: torch.Tensor,
        target_response: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Return normalized response predictions with shape [batch, future_time]."""
        if history.ndim != 3 or history.shape[-1] != 2:
            raise ValueError("history must have shape [batch, time, 2]")
        if future_command.ndim != 2:
            raise ValueError("future_command must have shape [batch, future_time]")
        hidden = None
        if self.encoder is not None:
            _, encoded = self.encoder(history)
            hidden = encoded[-1]
        previous_response = history[:, -1, 1]
        predictions = []
        for step in range(future_command.shape[1]):
            decoder_input = torch.stack([future_command[:, step], previous_response], dim=-1)
            if self.decoder_cell is not None:
                hidden = self.decoder_cell(decoder_input, hidden)
                delta = self.output(hidden).squeeze(-1)
            else:
                delta = self.output(decoder_input).squeeze(-1)
            prediction = previous_response + delta
            predictions.append(prediction)
            if target_response is not None and teacher_forcing_ratio > 0.0:
                use_teacher = torch.rand(prediction.shape[0], device=prediction.device) < teacher_forcing_ratio
                previous_response = torch.where(use_teacher, target_response[:, step], prediction)
            else:
                previous_response = prediction
        return torch.stack(predictions, dim=1)
