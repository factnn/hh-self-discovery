"""Independent numerical validation for the production HH simulator.

The reference equations below intentionally do not call the rate, current, or
integration routines in :mod:`hh_self_discovery.simulator`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ReferenceParameters:
    capacitance: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_l: float = -54.4


def _ratio(x: float, scale: float) -> float:
    q = x / scale
    if abs(q) < 1e-8:
        return scale * (1.0 + q / 2.0 + q * q / 12.0)
    return float(x / (-np.expm1(-q)))


def reference_rates(voltage: float) -> np.ndarray:
    """Independent implementation ordered as an,bn,am,bm,ah,bh."""
    v = float(voltage)
    return np.asarray(
        [
            0.01 * _ratio(v + 55.0, 10.0),
            0.125 * np.exp(-(v + 65.0) / 80.0),
            0.1 * _ratio(v + 40.0, 10.0),
            4.0 * np.exp(-(v + 65.0) / 18.0),
            0.07 * np.exp(-(v + 65.0) / 20.0),
            1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0)),
        ],
        dtype=np.float64,
    )


def reference_steady_state(voltage: float) -> np.ndarray:
    an, bn, am, bm, ah, bh = reference_rates(voltage)
    return np.asarray([an / (an + bn), am / (am + bm), ah / (ah + bh)])


def reference_gate_field(voltage: float, gates: np.ndarray) -> np.ndarray:
    an, bn, am, bm, ah, bh = reference_rates(voltage)
    n, m, h = gates
    return np.asarray(
        [an * (1.0 - n) - bn * n, am * (1.0 - m) - bm * m, ah * (1.0 - h) - bh * h]
    )


def reference_currents(
    voltage: np.ndarray | float,
    gates: np.ndarray,
    params: ReferenceParameters = ReferenceParameters(),
) -> np.ndarray:
    voltage = np.asarray(voltage, dtype=np.float64)
    gates = np.asarray(gates, dtype=np.float64)
    n, m, h = np.moveaxis(gates, -1, 0)
    return np.stack(
        [
            params.g_na * m**3 * h * (voltage - params.e_na),
            params.g_k * n**4 * (voltage - params.e_k),
            params.g_l * (voltage - params.e_l),
        ],
        axis=-1,
    )


def solve_reference(
    time_ms: np.ndarray,
    command: np.ndarray,
    mode: str,
    params: ReferenceParameters = ReferenceParameters(),
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Integrate an independently written HH system with SciPy DOP853."""
    time = np.asarray(time_ms, dtype=np.float64)
    command = np.asarray(command, dtype=np.float64)

    def control(t: float) -> float:
        return float(np.interp(t, time, command))

    if mode == "current_clamp":
        initial_voltage = -65.0
        y0 = np.concatenate(([initial_voltage], reference_steady_state(initial_voltage)))

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            voltage, n, m, h = y
            currents = reference_currents(voltage, np.asarray([n, m, h]), params)
            dv = (control(t) - float(currents.sum())) / params.capacitance
            return np.concatenate(([dv], reference_gate_field(voltage, y[1:])))

        solution = solve_ivp(
            rhs, (float(time[0]), float(time[-1])), y0, method="DOP853",
            t_eval=time, rtol=rtol, atol=atol, max_step=0.05,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        voltage = solution.y[0]
        gates = solution.y[1:].T
        current = command
    elif mode == "voltage_clamp":
        voltage = command
        y0 = reference_steady_state(float(voltage[0]))

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return reference_gate_field(control(t), y)

        solution = solve_ivp(
            rhs, (float(time[0]), float(time[-1])), y0, method="DOP853",
            t_eval=time, rtol=rtol, atol=atol, max_step=0.05,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        gates = solution.y.T
        current = reference_currents(voltage, gates, params).sum(axis=-1)
    else:
        raise ValueError(f"unknown mode: {mode}")
    components = reference_currents(voltage, gates, params)
    return {"voltage": voltage, "current": current, "gates": gates, "components": components}


def analytic_constant_voltage(
    time_ms: np.ndarray,
    voltage_mV: float,
    initial_gates: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form gate trajectory for a constant voltage command."""
    time = np.asarray(time_ms, dtype=np.float64)
    rates = reference_rates(voltage_mV).reshape(3, 2)
    total = rates.sum(axis=1)
    steady = rates[:, 0] / total
    initial = reference_steady_state(voltage_mV) if initial_gates is None else np.asarray(initial_gates)
    elapsed = time - time[0]
    return steady + (initial - steady) * np.exp(-elapsed[:, None] * total[None, :])


def error_summary(production, reference: dict[str, np.ndarray]) -> dict[str, float]:
    production_gates = np.stack([production.n, production.m, production.h], axis=-1)
    production_components = np.stack(
        [production.i_na_uA_cm2, production.i_k_uA_cm2, production.i_leak_uA_cm2], axis=-1
    )

    def rmse(a, b) -> float:
        return float(np.sqrt(np.mean(np.square(np.asarray(a) - np.asarray(b)))))

    def maximum(a, b) -> float:
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    return {
        "voltage_rmse_mV": rmse(production.voltage_mV, reference["voltage"]),
        "voltage_max_abs_mV": maximum(production.voltage_mV, reference["voltage"]),
        "current_rmse_uA_cm2": rmse(production.current_uA_cm2, reference["current"]),
        "current_max_abs_uA_cm2": maximum(production.current_uA_cm2, reference["current"]),
        "gate_rmse": rmse(production_gates, reference["gates"]),
        "gate_max_abs": maximum(production_gates, reference["gates"]),
        "component_current_rmse_uA_cm2": rmse(production_components, reference["components"]),
    }
