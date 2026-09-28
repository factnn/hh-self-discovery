import pytest
import torch

from hh_self_discovery.latent_training import (
    CURRENT_EVENT_PROFILES,
    curriculum_rollout_steps,
    multi_horizon_loss,
    spike_waveform_loss,
    conditional_spike_waveform_loss,
    train_structured_latent,
)
from hh_self_discovery.events import EventClass


def test_curriculum_can_reserve_epochs_at_full_horizon() -> None:
    values = [curriculum_rollout_steps(250, epoch, 30, 0.4) for epoch in range(30)]
    assert values[0] == 25
    assert values[11] == 250
    assert all(value == 250 for value in values[11:])


def test_curriculum_fraction_is_validated() -> None:
    with pytest.raises(ValueError):
        curriculum_rollout_steps(250, 0, 10, 0.0)


def test_multi_horizon_loss_is_differentiable_and_zero_when_exact() -> None:
    target = torch.randn(4, 20)
    activity = torch.rand(4, 20)
    exact = target.clone().requires_grad_(True)
    exact_loss = multi_horizon_loss(exact, target, activity)
    torch.testing.assert_close(exact_loss, torch.zeros_like(exact_loss))
    perturbed = (target + 0.2).requires_grad_(True)
    loss = multi_horizon_loss(perturbed, target, activity)
    assert loss > 0
    loss.backward()
    assert perturbed.grad is not None


def test_encoder_size_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError, match="encoder_size"):
        train_structured_latent("unused", str(tmp_path), 3, encoder_size=0)


def test_early_stopping_parameters_reject_negative_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="early_stopping_patience"):
        train_structured_latent("unused", str(tmp_path), 3, early_stopping_patience=-1)
    with pytest.raises(ValueError, match="early_stopping_min_delta"):
        train_structured_latent("unused", str(tmp_path), 3, early_stopping_min_delta=-0.1)
    with pytest.raises(ValueError, match="current_clamp_loss_weight"):
        train_structured_latent("unused", str(tmp_path), 3, current_clamp_loss_weight=0.0)


def test_spike_focused_profile_prioritizes_active_phases() -> None:
    fractions = CURRENT_EVENT_PROFILES["spike_focused"]
    assert fractions is not None
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert fractions[EventClass.ACTIVE] == 0.40
    assert fractions[EventClass.RECOVERY] == 0.20


def test_unknown_event_profile_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown current_event_profile"):
        train_structured_latent(
            "unused",
            str(tmp_path),
            3,
            current_event_profile="not_a_profile",
        )


def test_spike_waveform_loss_prefers_matching_waveform():
    target = torch.tensor([[-2.0, -1.0, 1.5, -0.5, -2.0]])
    matching = target.clone().requires_grad_(True)
    flat = torch.full_like(target, -2.0)
    matching_loss = spike_waveform_loss(matching, target, threshold=0.0)
    flat_loss = spike_waveform_loss(flat, target, threshold=0.0)
    assert matching_loss < flat_loss
    matching_loss.backward()
    assert matching.grad is not None


def test_conditional_spike_loss_handles_mixed_windows():
    target = torch.tensor([[-2.0, -1.5, -1.0], [-2.0, 1.5, -2.0]])
    matching = target.clone().requires_grad_(True)
    loss = conditional_spike_waveform_loss(matching, target, threshold=0.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert matching.grad is not None
