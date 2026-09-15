from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml.energy.rcsd1yd_failure_analysis import (
    calibration_boundaries,
    group_summary,
    monthly_route_summary,
    probability_calibration_table,
    raw_auc_ap,
    spearman_rank_correlation,
    threshold_diagnostics,
)


def _config() -> dict:
    return yaml.safe_load(
        Path("configs/rcsd1yd_router_failure_analysis.yaml")
        .read_text(encoding="utf-8")
    )


def test_protocol_is_post_hoc_and_noncausal() -> None:
    config = _config()

    assert config["study"]["evidence_class"] == "POST_HOC_EXPLORATORY"
    assert config["study"]["causal_claim"] is False
    assert config["study"]["closed_result_may_change"] is False
    assert all(bool(value) for value in config["forbidden"].values())


def test_raw_auc_ap_keeps_raw_direction() -> None:
    y = np.asarray([False, False, True, True])
    score = np.asarray([4.0, 3.0, 2.0, 1.0])

    result = raw_auc_ap(score, y)

    assert result["roc_auc"] == pytest.approx(0.0)


def test_spearman_rank_correlation() -> None:
    x = np.asarray([1.0, 2.0, 3.0, 4.0])
    y = np.asarray([10.0, 20.0, 30.0, 40.0])

    assert spearman_rank_correlation(x, y) == pytest.approx(1.0)


def test_calibration_boundaries_are_applied_unchanged() -> None:
    cal = np.arange(10, dtype=float) / 10.0
    boundaries = calibration_boundaries(
        cal,
        quantiles=[0.25, 0.50, 0.75],
    )

    evaluation = np.asarray([0.05, 0.55, 0.95])
    high = np.asarray([False, True, True])

    table = probability_calibration_table(
        evaluation,
        high,
        boundaries=boundaries,
    )

    assert len(table) == 4
    assert sum(int(row["rows"]) for row in table) == 3


def test_threshold_enrichment() -> None:
    probability = np.asarray([0.1, 0.2, 0.8, 0.9])
    high = np.asarray([False, False, True, True])

    result = threshold_diagnostics(
        probability,
        high,
        threshold=0.5,
    )

    assert result["invocation_fraction"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["high_evidence_enrichment"] == pytest.approx(2.0)


def test_group_summary() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )
    probability = np.asarray([0.1, 0.2, 0.3])
    pca = pd.Series([10.0, 20.0, 30.0])
    tcn = np.asarray([0.5, 0.6, 0.7])
    mask = np.asarray([False, True, True])

    result = group_summary(
        mask=mask,
        router_probability=probability,
        router_features=frame,
        pca_ewma=pca,
        raw_tcn_score=tcn,
    )

    assert result["count"] == 2
    assert result["features"]["a"]["median"] == pytest.approx(2.5)


def test_monthly_summary() -> None:
    index = pd.DatetimeIndex(
        [
            "2022-10-01",
            "2022-10-02",
            "2022-11-01",
            "2022-11-02",
        ]
    )
    high = np.asarray([True, False, True, True])
    route = np.asarray([True, False, False, True])
    q90 = np.asarray([True, True, True, True])

    result = monthly_route_summary(
        target_index=index,
        high_evidence=high,
        evidence_route=route,
        q90_route=q90,
        months=["2022-10", "2022-11"],
    )

    assert result["2022-10"]["valid_tcn_rows"] == 2
    assert result["2022-10"]["evidence_aware_high_coverage"] == pytest.approx(
        1.0
    )
    assert result["2022-11"]["evidence_aware_high_coverage"] == pytest.approx(
        0.5
    )
