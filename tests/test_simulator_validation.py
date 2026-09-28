import numpy as np

from hh_self_discovery import current_step, simulate_current_clamp, simulate_voltage_clamp, time_grid
from hh_self_discovery.simulator_validation import (
    analytic_constant_voltage,
    error_summary,
    reference_steady_state,
    solve_reference,
)


def test_independent_dop853_matches_current_clamp() -> None:
    time = time_grid(20.0, 0.02)
    command = current_step(time, 8.0, 5.0, 15.0)
    production = simulate_current_clamp(time, command)
    metrics = error_summary(production, solve_reference(time, command, "current_clamp"))
    assert metrics["voltage_rmse_mV"] < 0.1
    assert metrics["gate_rmse"] < 1e-3


def test_constant_voltage_matches_closed_form() -> None:
    time = time_grid(20.0, 0.02)
    initial = reference_steady_state(-65.0)
    voltage = -20.0
    production = simulate_voltage_clamp(
        time, np.full_like(time, voltage), initial_gates=initial, include_capacitive_current=False
    )
    expected = analytic_constant_voltage(time, voltage, initial)
    actual = np.stack([production.n, production.m, production.h], axis=-1)
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
