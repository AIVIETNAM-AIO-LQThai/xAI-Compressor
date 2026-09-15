from __future__ import annotations

import pandas as pd
import yaml

from scripts.fit_cobra_train_models import (
    CONFIG_PATH,
    expected_sequence_count,
    validate_frozen_config,
)


def test_frozen_fit_config_is_valid() -> None:
    config = yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )
    validate_frozen_config(config)


def test_sequence_count_respects_segments() -> None:
    segments = pd.Series(
        [0] * 20
        + [1] * 12
        + [2] * 15
    )

    assert expected_sequence_count(
        segments,
        history_bins=12,
    ) == 11


def test_train_csv_loader_preserves_float_round_trip(
    tmp_path,
) -> None:
    import numpy as np

    from scripts.fit_cobra_train_models import (
        load_train_representation,
    )

    original = pd.DataFrame(
        {
            "day": ["2024-06-03"],
            "timestamp": pd.to_datetime(
                ["2024-06-03 10:00:00"]
            ),
            "segment_id": [0],
            "sensor": [
                np.float64(
                    0.12345678901234568
                )
            ],
        }
    )

    path = tmp_path / "train.csv"

    original.to_csv(
        path,
        index=False,
        float_format="%.17g",
    )

    loaded = load_train_representation(
        path
    )

    assert (
        loaded["sensor"]
        .to_numpy(dtype=np.float64)
        .view(np.uint64)[0]
        ==
        original["sensor"]
        .to_numpy(dtype=np.float64)
        .view(np.uint64)[0]
    )
