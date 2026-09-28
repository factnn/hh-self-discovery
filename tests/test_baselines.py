import numpy as np

from hh_self_discovery.metrics import regression_metrics, waveform_metrics


def test_regression_metrics_detect_flat_line_derivative_failure() -> None:
    target = np.asarray([[0.0, 1.0, 3.0], [2.0, 2.0, 1.0]])
    prediction = np.asarray([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    activity = np.asarray([[0.0, 0.5, 1.0], [0.0, 0.0, 1.0]])
    metrics = regression_metrics(target, prediction, activity)
    assert metrics["rmse"] > 0.0
    assert metrics["derivative_rmse"] > 0.0
    assert metrics["activity_weighted_rmse"] > metrics["rmse"]


def test_perfect_prediction_has_zero_errors() -> None:
    target = np.arange(12, dtype=float).reshape(3, 4)
    metrics = regression_metrics(target, target.copy(), np.zeros_like(target))
    assert metrics["mse"] == 0.0
    assert metrics["derivative_rmse"] == 0.0


def test_waveform_metrics_are_zero_for_perfect_spikes() -> None:
    trace = np.asarray([[-65.0, -20.0, 20.0, -30.0, -60.0]])
    metrics = waveform_metrics(trace, trace.copy(), "current_clamp", 0.1)
    assert metrics["spike_count_mae"] == 0.0
    assert metrics["first_spike_timing_mae_ms"] == 0.0
    assert metrics["repolarization_timing_mae_ms"] == 0.0
