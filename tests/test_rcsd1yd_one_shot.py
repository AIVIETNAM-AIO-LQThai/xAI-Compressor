from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml.energy.rcsd1yd_evaluation import (
    ROUTE_NAMES,
    evidence_stage_status,
    final_external_status,
    frozen_router_probability,
    gap_aware_router_features,
)


def _study() -> dict:
    return yaml.safe_load(
        Path("configs/rcsd1yd_external_evaluation.yaml")
        .read_text(encoding="utf-8")
    )


def _execution() -> dict:
    return yaml.safe_load(
        Path("configs/rcsd1yd_one_shot_execution.yaml")
        .read_text(encoding="utf-8")
    )


def test_execution_systems_match_frozen_protocol() -> None:
    study = _study()
    execution = _execution()

    assert tuple(study["evaluation"]["systems"]) == ROUTE_NAMES
    assert (
        tuple(execution["energy_execution"]["conditions"])
        == ROUTE_NAMES
    )


def test_zero_denominator_is_inconclusive() -> None:
    metrics = {
        "high_evidence_bins": 0,
        "tcn_alert_bins": 5,
        "high_evidence_coverage": None,
        "tcn_alert_bin_coverage": 1.0,
    }

    assert (
        evidence_stage_status(
            metrics,
            minimum_high_coverage=0.8,
            minimum_alert_coverage=0.8,
        )
        == "INCONCLUSIVE_EXTERNAL_EVALUATION"
    )


def test_final_external_status_requires_all_criteria() -> None:
    criteria = _study()["success_criteria"]
    metrics = {
        "high_evidence_coverage": 0.90,
        "tcn_alert_bin_coverage": 0.85,
    }

    status = final_external_status(
        evidence_status="EVIDENCE_COVERAGE_PASS",
        evidence_aware_metrics=metrics,
        mean_gross_energy_reduction=0.60,
        gross_reductions=[0.55, 0.61, 0.58, 0.62, 0.64],
        success_criteria=criteria,
    )
    assert status == "EXTERNAL_TRANSPORT_PASS"

    failed = final_external_status(
        evidence_status="EVIDENCE_COVERAGE_PASS",
        evidence_aware_metrics=metrics,
        mean_gross_energy_reduction=0.49,
        gross_reductions=[0.55, 0.61, 0.58, 0.62, 0.64],
        success_criteria=criteria,
    )
    assert failed == "EXTERNAL_TRANSPORT_FAIL"


def test_frozen_router_probability() -> None:
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0],
            "b": [0.0, 0.0],
        }
    )
    frozen = {
        "feature_names": ["a", "b"],
        "feature_scaler": {
            "mean": [0.0, 0.0],
            "scale": [1.0, 1.0],
        },
        "logistic": {
            "coef": [[1.0, 0.0]],
            "intercept": [0.0],
        },
    }

    result = frozen_router_probability(frame, frozen)
    assert result[0] == pytest.approx(0.5)
    assert result[1] > result[0]


def test_gap_aware_router_history_does_not_bridge_gap() -> None:
    index = pd.DatetimeIndex(
        [
            "2022-01-01 00:00",
            "2022-01-01 00:15",
            "2022-01-01 00:30",
            "2022-01-01 00:45",
            "2022-01-01 01:00",
            "2022-01-01 02:00",
            "2022-01-01 02:15",
            "2022-01-01 02:30",
            "2022-01-01 02:45",
            "2022-01-01 03:00",
        ]
    )
    values = pd.Series(
        np.arange(len(index), dtype=float) + 1.0,
        index=index,
    )
    references = {
        "pca_ewma_reference": list(np.linspace(0.0, 20.0, 20)),
        "volatility_reference": list(np.linspace(0.0, 10.0, 20)),
        "feature_support_reference": list(np.linspace(0.0, 20.0, 20)),
        "latent_support_reference": list(np.linspace(0.0, 20.0, 20)),
        "pca_q90_threshold": 5.0,
        "pca_q95_threshold": 6.0,
        "delta_center_median": 1.0,
        "delta_scale_iqr": 1.0,
    }

    target_index = pd.DatetimeIndex([index[4], index[9]])

    frame = gap_aware_router_features(
        pca_ewma=values,
        feature_support_raw=values,
        latent_support_raw=values,
        target_index=target_index,
        references=references,
        rolling_history_bins=5,
        reset_gap_minutes=30,
    )

    assert len(frame) == 2
    assert np.isfinite(frame.to_numpy(dtype=float)).all()
    # The second target uses only the five post-gap rows.
    assert frame.loc[index[9], "recent_q90_hit_fraction"] == pytest.approx(
        1.0
    )
