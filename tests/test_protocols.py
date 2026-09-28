import numpy as np
import pytest

from hh_self_discovery.protocols import (
    current_step,
    smooth_random_current,
    time_grid,
    voltage_step,
)


def test_time_grid_is_inclusive_and_uniform() -> None:
    time = time_grid(1.0, 0.1)
    assert len(time) == 11
    assert time[0] == 0.0
    assert time[-1] == 1.0
    np.testing.assert_allclose(np.diff(time), 0.1)


def test_time_grid_rejects_incommensurate_duration() -> None:
    with pytest.raises(ValueError):
        time_grid(1.0, 0.3)


def test_step_protocol_boundaries() -> None:
    time = time_grid(10.0, 1.0)
    current = current_step(time, 5.0, 2.0, 5.0)
    voltage = voltage_step(time, -80.0, 0.0, 2.0, 5.0, tail_mV=-60.0)
    np.testing.assert_array_equal(current, [0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(voltage, [-80, -80, 0, 0, 0, -60, -60, -60, -60, -60, -60])


def test_smooth_random_current_is_reproducible_and_correlated() -> None:
    time = time_grid(20.0, 0.1)
    first = smooth_random_current(time, np.random.default_rng(4))
    second = smooth_random_current(time, np.random.default_rng(4))
    np.testing.assert_array_equal(first, second)
    assert np.corrcoef(first[:-1], first[1:])[0, 1] > 0.8
