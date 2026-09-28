import torch

from hh_self_discovery.latent_model import StructuredLatentHH


def _normalizers() -> dict:
    return {
        "current_clamp": {
            "command": {"mean": 2.0, "std": 5.0},
            "response": {"mean": -55.0, "std": 25.0},
        },
        "voltage_clamp": {
            "command": {"mean": -60.0, "std": 30.0},
            "response": {"mean": 10.0, "std": 300.0},
        },
    }


def test_structured_model_shapes_bounds_and_gradients() -> None:
    model = StructuredLatentHH(3, _normalizers(), encoder_size=16)
    for mode in model.MODES:
        history = torch.randn(4, 20, 2)
        command = torch.randn(4, 8)
        prediction, latents = model(history, command, mode, return_latents=True)
        assert prediction.shape == (4, 8)
        assert latents.shape == (4, 8, 3)
        assert torch.all((latents >= 0.0) & (latents <= 1.0))
        (prediction.square().mean() + model.complexity_penalty()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_explicit_state_rollout_matches_forward() -> None:
    torch.manual_seed(7)
    model = StructuredLatentHH(3, _normalizers(), encoder_size=16)
    history = torch.randn(2, 20, 2)
    command = torch.randn(2, 8)
    for mode in model.MODES:
        expected, expected_latents = model(history, command, mode, return_latents=True)
        gates = model.encode(history, mode)
        previous = model.denormalize(history[:, -1, 1], mode, "response")
        actual, actual_latents = model.rollout_from_state(
            gates,
            previous,
            command,
            mode,
            return_latents=True,
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_latents, expected_latents)


def test_current_parameters_are_physical() -> None:
    model = StructuredLatentHH(2, _normalizers())
    params = model.current_parameters()
    assert torch.all(params["conductance"] > 0.0)
    assert torch.all(params["exponents"] > 0.0)
    assert torch.all((-120.0 < params["reversal_mV"]) & (params["reversal_mV"] < 100.0))
