import pytest
import torch

from hh_self_discovery.baseline_models import build_baseline_model


@pytest.mark.parametrize("model_type", ["gru_bottleneck"])
def test_baseline_shapes_and_latents(model_type):
    model = build_baseline_model(model_type, latent_size=3, encoder_width=8, dynamics_width=8)
    history = torch.randn(2, 12, 2)
    command = torch.randn(2, 5)
    prediction, latent = model(history, command, "current_clamp", return_latents=True)
    assert prediction.shape == (2, 5)
    assert latent.shape == (2, 5, 3)
    assert torch.isfinite(prediction).all()


def test_unknown_baseline_rejected():
    with pytest.raises(ValueError):
        build_baseline_model("not-a-model", latent_size=3)
