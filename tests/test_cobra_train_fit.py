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
