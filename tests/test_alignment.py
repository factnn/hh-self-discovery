import numpy as np

from hh_self_discovery.alignment import affine_positive_control, align_pair


def _bundle(state: np.ndarray, field: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "state": state,
        "field": field,
        "prediction": prediction,
        "target": prediction,
        "mode": np.arange(len(state), dtype=np.int64) % 2,
        "group": np.arange(len(state)),
    }


def test_affine_alignment_recovers_known_coordinate_change() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(1000, 3))
    source_field = rng.normal(size=(1000, 3))
    matrix = np.asarray([[1.2, 0.2, 0.0], [0.1, 0.8, -0.1], [0.0, 0.3, 1.1]])
    offset = np.asarray([0.2, -0.4, 0.1])
    target = source @ matrix + offset
    target_field = source_field @ matrix
    prediction = rng.normal(size=1000)
    validation = slice(0, 600)
    test = slice(600, None)
    result = align_pair(
        _bundle(source[validation], source_field[validation], prediction[validation]),
        _bundle(target[validation], target_field[validation], prediction[validation]),
        _bundle(source[test], source_field[test], prediction[test]),
        _bundle(target[test], target_field[test], prediction[test]),
        families=("affine",),
    )["maps"]["affine"]
    assert result["mean_test_r2"] > 0.999999
    assert result["mean_cycle_r2"] > 0.999999
    assert result["vector_field_relative_rmse"] < 1e-5


def test_positive_control_passes_conjugacy_metrics() -> None:
    rng = np.random.default_rng(9)
    validation_state = rng.uniform(0.0, 1.0, size=(500, 3))
    test_state = rng.uniform(0.0, 1.0, size=(300, 3))
    validation_field = rng.normal(scale=0.2, size=(500, 3))
    test_field = rng.normal(scale=0.2, size=(300, 3))
    result = affine_positive_control(
        validation_state, test_state, validation_field, test_field
    )
    assert result["mean_test_r2"] > 0.999999
    assert result["mean_cycle_r2"] > 0.999999
    assert result["vector_field_relative_rmse"] < 1e-5
