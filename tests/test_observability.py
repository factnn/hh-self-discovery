import numpy as np
import pytest

from hh_self_discovery.observability import gramian_metrics


def test_gramian_recovers_full_rank_and_condition() -> None:
    metrics = gramian_metrics(np.eye(3))
    assert metrics["effective_rank"] == 3
    assert metrics["condition_number"] == pytest.approx(1.0)
    assert metrics["minimum_eigenvalue"] == pytest.approx(1.0 / 3.0)


def test_gramian_detects_unobservable_direction() -> None:
    jacobian = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    metrics = gramian_metrics(jacobian)
    assert metrics["effective_rank"] == 2
    assert metrics["minimum_eigenvalue"] == 0.0
