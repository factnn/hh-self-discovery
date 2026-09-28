import numpy as np

from hh_self_discovery.ood_dataset import OOD_FAMILIES, sample_ood_protocol


def test_ood_families_are_finite_and_use_both_clamp_modes() -> None:
    modes = set()
    for index, family in enumerate(OOD_FAMILIES):
        trace, metadata = sample_ood_protocol(family, np.random.default_rng(index), 0.05)
        assert np.all(np.isfinite(trace.current_uA_cm2))
        assert np.all(np.isfinite(trace.voltage_mV))
        assert metadata["ood"] is True
        modes.add(trace.control_mode)
    assert modes == {"current_clamp", "voltage_clamp"}
