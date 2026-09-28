"""HH latent-state discovery package."""

from .protocols import current_step, time_grid, voltage_step
from .simulator import HHParameters, Trajectory, simulate_current_clamp, simulate_voltage_clamp

__all__ = [
    "HHParameters",
    "Trajectory",
    "current_step",
    "simulate_current_clamp",
    "simulate_voltage_clamp",
    "time_grid",
    "voltage_step",
]

