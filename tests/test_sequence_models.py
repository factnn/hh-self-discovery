import numpy as np
import torch

from hh_self_discovery.sequence_models import AutoregressiveSequenceModel
from hh_self_discovery.training import ModeNormalizers, Standardizer


def test_rnn_and_gru_output_shapes_and_gradients() -> None:
    history = torch.randn(4, 20, 2)
    command = torch.randn(4, 7)
    target = torch.randn(4, 7)
    for cell_type in ("mlp", "rnn", "gru"):
        model = AutoregressiveSequenceModel(cell_type, hidden_size=8)
        prediction = model(history, command, target, teacher_forcing_ratio=0.5)
        assert prediction.shape == target.shape
        prediction.square().mean().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_models_start_as_persistence() -> None:
    history = torch.randn(3, 12, 2)
    command = torch.randn(3, 5)
    for cell_type in ("mlp", "rnn", "gru"):
        model = AutoregressiveSequenceModel(cell_type, hidden_size=8)
        prediction = model(history, command)
        expected = history[:, -1:, 1].expand_as(prediction)
        torch.testing.assert_close(prediction, expected)


def test_standardizers_round_trip() -> None:
    normalizers = ModeNormalizers(Standardizer(2.0, 3.0), Standardizer(-4.0, 5.0))
    values = np.asarray([-2.0, 0.0, 7.0])
    np.testing.assert_allclose(normalizers.command.inverse(normalizers.command.transform(values)), values)
    np.testing.assert_allclose(normalizers.response.inverse(normalizers.response.transform(values)), values)
