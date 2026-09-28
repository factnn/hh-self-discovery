"""Numerically explicit Hodgkin-Huxley simulator for intervention datasets.

Sign convention: positive membrane current is outward. In current clamp,
``C dV/dt = I_applied - I_ionic``. In voltage clamp, the required command
current is ``I_clamp = C dV_command/dt + I_ionic``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HHParameters:
    capacitance_uF_cm2: float = 1.0
    g_na_mS_cm2: float = 120.0
    g_k_mS_cm2: float = 36.0
    g_leak_mS_cm2: float = 0.3
    e_na_mV: float = 50.0
    e_k_mV: float = -77.0
    e_leak_mV: float = -54.4


@dataclass(frozen=True)
class Trajectory:
    time_ms: np.ndarray
    current_uA_cm2: np.ndarray
    voltage_mV: np.ndarray
    n: np.ndarray
    m: np.ndarray
    h: np.ndarray
    i_na_uA_cm2: np.ndarray
    i_k_uA_cm2: np.ndarray
    i_leak_uA_cm2: np.ndarray
    control_mode: str

    def observed(self) -> dict[str, np.ndarray | str]:
        """Return the only fields that a discovery model may consume."""
        return {
            "time_ms": self.time_ms,
            "current_uA_cm2": self.current_uA_cm2,
            "voltage_mV": self.voltage_mV,
            "control_mode": self.control_mode,
        }


def _safe_exp(value: float | np.ndarray) -> float | np.ndarray:
    return np.exp(np.clip(value, -700.0, 700.0))


def _vtrap_ratio(x: float, scale: float) -> float:
    """Evaluate x / (1 - exp(-x/scale)) near its removable singularity."""
    ratio = x / scale
    if abs(ratio) < 1e-7:
        return scale * (1.0 + ratio / 2.0 + ratio * ratio / 12.0)
    return float(x / (-np.expm1(-ratio)))


def gate_rates(voltage_mV: float) -> tuple[float, float, float, float, float, float]:
    """Return alpha/beta rates for n, m, and h in inverse milliseconds."""
    v = float(voltage_mV)
    alpha_n = 0.01 * _vtrap_ratio(v + 55.0, 10.0)
    beta_n = 0.125 * _safe_exp(-(v + 65.0) / 80.0)
    alpha_m = 0.1 * _vtrap_ratio(v + 40.0, 10.0)
    beta_m = 4.0 * _safe_exp(-(v + 65.0) / 18.0)
    alpha_h = 0.07 * _safe_exp(-(v + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + _safe_exp(-(v + 35.0) / 10.0))
    return tuple(float(x) for x in (alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h))


def steady_state_gates(voltage_mV: float) -> np.ndarray:
    """Return the equilibrium state [n, m, h] at a fixed voltage."""
    an, bn, am, bm, ah, bh = gate_rates(voltage_mV)
    return np.asarray([an / (an + bn), am / (am + bm), ah / (ah + bh)], dtype=np.float64)


def gate_derivatives(voltage_mV: float, gates: np.ndarray) -> np.ndarray:
    an, bn, am, bm, ah, bh = gate_rates(voltage_mV)
    n, m, h = gates
    return np.asarray(
        [an * (1.0 - n) - bn * n, am * (1.0 - m) - bm * m, ah * (1.0 - h) - bh * h],
        dtype=np.float64,
    )


def ionic_currents(
    voltage_mV: float | np.ndarray,
    n: float | np.ndarray,
    m: float | np.ndarray,
    h: float | np.ndarray,
    params: HHParameters = HHParameters(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return outward sodium, potassium, and leak current densities."""
    voltage = np.asarray(voltage_mV, dtype=np.float64)
    i_na = params.g_na_mS_cm2 * np.asarray(m) ** 3 * np.asarray(h) * (voltage - params.e_na_mV)
    i_k = params.g_k_mS_cm2 * np.asarray(n) ** 4 * (voltage - params.e_k_mV)
    i_leak = params.g_leak_mS_cm2 * (voltage - params.e_leak_mV)
    return i_na, i_k, i_leak


def _validate_series(time_ms: np.ndarray, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time_ms, dtype=np.float64)
    values = np.asarray(command, dtype=np.float64)
    if time.ndim != 1 or values.shape != time.shape or len(time) < 2:
        raise ValueError("time and command must be one-dimensional arrays with equal length >= 2")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        raise ValueError("time and command must contain finite values")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing")
    return time, values


def _rk4_step(state: np.ndarray, dt: float, derivative) -> np.ndarray:
    k1 = derivative(0.0, state)
    k2 = derivative(0.5 * dt, state + 0.5 * dt * k1)
    k3 = derivative(0.5 * dt, state + 0.5 * dt * k2)
    k4 = derivative(dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_current_clamp(
    time_ms: np.ndarray,
    applied_current_uA_cm2: np.ndarray,
    params: HHParameters = HHParameters(),
    initial_voltage_mV: float = -65.0,
    initial_gates: np.ndarray | None = None,
) -> Trajectory:
    """Integrate voltage and gates for a prescribed applied-current waveform."""
    time, current = _validate_series(time_ms, applied_current_uA_cm2)
    state = np.empty((len(time), 4), dtype=np.float64)
    gates0 = steady_state_gates(initial_voltage_mV) if initial_gates is None else np.asarray(initial_gates, dtype=np.float64)
    if gates0.shape != (3,):
        raise ValueError("initial_gates must have shape (3,) ordered as n, m, h")
    state[0] = [initial_voltage_mV, *gates0]

    for index in range(len(time) - 1):
        t0 = time[index]
        dt = time[index + 1] - t0

        def derivative(offset: float, y: np.ndarray) -> np.ndarray:
            voltage, n, m, h = y
            input_current = float(np.interp(t0 + offset, time[index : index + 2], current[index : index + 2]))
            ionic = sum(float(x) for x in ionic_currents(voltage, n, m, h, params))
            dv = (input_current - ionic) / params.capacitance_uF_cm2
            return np.concatenate(([dv], gate_derivatives(voltage, np.asarray([n, m, h]))))

        state[index + 1] = _rk4_step(state[index], dt, derivative)
        state[index + 1, 1:] = np.clip(state[index + 1, 1:], 0.0, 1.0)

    voltage, n, m, h = state.T
    i_na, i_k, i_leak = ionic_currents(voltage, n, m, h, params)
    return Trajectory(time, current, voltage, n, m, h, i_na, i_k, i_leak, "current_clamp")


def simulate_voltage_clamp(
    time_ms: np.ndarray,
    voltage_command_mV: np.ndarray,
    params: HHParameters = HHParameters(),
    initial_gates: np.ndarray | None = None,
    include_capacitive_current: bool = True,
) -> Trajectory:
    """Integrate gates under a voltage command and return required clamp current."""
    time, voltage = _validate_series(time_ms, voltage_command_mV)
    gates = np.empty((len(time), 3), dtype=np.float64)
    gates[0] = steady_state_gates(voltage[0]) if initial_gates is None else np.asarray(initial_gates, dtype=np.float64)
    if gates[0].shape != (3,):
        raise ValueError("initial_gates must have shape (3,) ordered as n, m, h")

    for index in range(len(time) - 1):
        t0 = time[index]
        dt = time[index + 1] - t0

        def derivative(offset: float, y: np.ndarray) -> np.ndarray:
            command_voltage = float(np.interp(t0 + offset, time[index : index + 2], voltage[index : index + 2]))
            return gate_derivatives(command_voltage, y)

        gates[index + 1] = np.clip(_rk4_step(gates[index], dt, derivative), 0.0, 1.0)

    n, m, h = gates.T
    i_na, i_k, i_leak = ionic_currents(voltage, n, m, h, params)
    ionic_total = i_na + i_k + i_leak
    if include_capacitive_current:
        edge_order = 2 if len(time) >= 3 else 1
        dv_dt = np.gradient(voltage, time, edge_order=edge_order)
        clamp_current = ionic_total + params.capacitance_uF_cm2 * dv_dt
    else:
        clamp_current = ionic_total
    return Trajectory(time, clamp_current, voltage, n, m, h, i_na, i_k, i_leak, "voltage_clamp")

