import json
from pathlib import Path

import numpy as np

from hh_self_discovery.parameter_dataset import generate_parameter_dataset


def test_parameter_dataset_is_evaluation_only_split(tmp_path: Path) -> None:
    root = tmp_path / "parameter_ood"
    records = generate_parameter_dataset(root, count=5, dt_ms=0.02, workers=1)
    assert all(record["split"] == "test" and record["parameter_ood"] for record in records)
    assert len({json.dumps(record["hh_parameters"], sort_keys=True) for record in records}) == 5
    with np.load(root / records[0]["observed_path"]) as archive:
        assert not {"n", "m", "h"}.intersection(archive.files)
