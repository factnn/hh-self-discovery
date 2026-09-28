import numpy as np

from hh_self_discovery import current_step, simulate_current_clamp, simulate_voltage_clamp, time_grid, voltage_step
from hh_self_discovery.simulator import (
    HHParameters,
    gate_derivatives,
    gate_rates,
    ionic_currents,
    steady_state_gates,
)


def test_rate_functions_are_finite_at_removable_singularities() -> None:
    for voltage in (-55.0, -40.0):
        rates = gate_rates(voltage)
        assert np.all(np.isfinite(rates))
        assert np.all(np.asarray(rates) > 0.0)


def test_steady_state_gates_are_fixed_points() -> None:
    for voltage in (-100.0, -65.0, -20.0, 40.0):
        gates = steady_state_gates(voltage)
        assert np.all((gates >= 0.0) & (gates <= 1.0))
        np.testing.assert_allclose(gate_derivatives(voltage, gates), 0.0, atol=1e-12)


def test_resting_current_clamp_remains_near_rest() -> None:
    time = time_grid(20.0, 0.01)
    trace = simulate_current_clamp(time, np.zeros_like(time))
    assert np.max(np.abs(trace.voltage_mV + 65.0)) < 0.1
    assert set(trace.observed()) == {"time_ms", "current_uA_cm2", "voltage_mV", "control_mode"}


def test_current_step_generates_an_action_potential() -> None:
    time = time_grid(60.0, 0.01)
    current = current_step(time, 10.0, 5.0, 35.0)
    trace = simulate_current_clamp(time, current)
    assert trace.voltage_mV.max() > 0.0
    assert np.all((trace.n >= 0.0) & (trace.n <= 1.0))
    assert np.all((trace.m >= 0.0) & (trace.m <= 1.0))
    assert np.all((trace.h >= 0.0) & (trace.h <= 1.0))


def test_current_clamp_converges_when_timestep_is_halved() -> None:
    coarse_time = time_grid(40.0, 0.02)
    fine_time = time_grid(40.0, 0.01)
    coarse = simulate_current_clamp(coarse_time, current_step(coarse_time, 10.0, 5.0, 25.0))
    fine = simulate_current_clamp(fine_time, current_step(fine_time, 10.0, 5.0, 25.0))
    fine_on_coarse = np.interp(coarse_time, fine_time, fine.voltage_mV)
    assert np.sqrt(np.mean((coarse.voltage_mV - fine_on_coarse) ** 2)) < 0.25


def test_voltage_clamp_relaxes_toward_command_steady_state() -> None:
    time = time_grid(60.0, 0.01)
    command = voltage_step(time, -80.0, 0.0, 5.0, 55.0)
    trace = simulate_voltage_clamp(time, command, include_capacitive_current=False)
    expected = steady_state_gates(0.0)
    before_step_end = np.searchsorted(time, 55.0) - 1
    np.testing.assert_allclose(
        [trace.n[before_step_end], trace.m[before_step_end], trace.h[before_step_end]],
        expected,
        atol=2e-3,
    )
    assert trace.control_mode == "voltage_clamp"


def test_voltage_clamp_converges_when_timestep_is_halved() -> None:
    coarse_time = time_grid(30.0, 0.02)
    fine_time = time_grid(30.0, 0.01)
    coarse_command = -65.0 + 30.0 * np.sin(2.0 * np.pi * coarse_time / 30.0)
    fine_command = -65.0 + 30.0 * np.sin(2.0 * np.pi * fine_time / 30.0)
    coarse = simulate_voltage_clamp(
        coarse_time, coarse_command, include_capacitive_current=False
    )
    fine = simulate_voltage_clamp(
        fine_time, fine_command, include_capacitive_current=False
    )
    fine_current = np.interp(coarse_time, fine_time, fine.current_uA_cm2)
    assert np.sqrt(np.mean((coarse.current_uA_cm2 - fine_current) ** 2)) < 0.05


def test_each_ionic_component_vanishes_at_its_reversal_potential() -> None:
    params = HHParameters()
    n, m, h = 0.4, 0.3, 0.6
    i_na, _, _ = ionic_currents(params.e_na_mV, n, m, h, params)
    _, i_k, _ = ionic_currents(params.e_k_mV, n, m, h, params)
    _, _, i_leak = ionic_currents(params.e_leak_mV, n, m, h, params)
    np.testing.assert_allclose([i_na, i_k, i_leak], 0.0, atol=1e-12)
