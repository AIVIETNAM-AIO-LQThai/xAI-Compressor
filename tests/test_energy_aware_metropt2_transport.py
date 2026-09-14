from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_transport_config_preserves_primary_method() -> None:
    transport = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "energy_aware_metropt2_transport.yaml"
        ).read_text(encoding="utf-8")
    )
    study = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "energy_aware_hierarchical_intelligence.yaml"
        ).read_text(encoding="utf-8")
    )
    benchmark = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "energy_aware_inference_benchmark.yaml"
        ).read_text(encoding="utf-8")
    )

    assert transport["transport"]["retuning_allowed"] is False
    assert transport["transport"]["independent_confirmation"] is False

    feature = transport["feature_schema"]
    assert feature["variant"] == "flowmeter_excluded"
    assert feature["excluded_groups"] == ["flowmeter"]
    assert feature["expected_feature_count"] == 63
    assert feature["require_primary_feature_name_and_order_identity"] is True

    method = transport["method"]
    assert method["tcn"]["scaler"] == study["temporal_model"]["scaler"]
    assert method["tcn"]["scaler"] == "standard"
    assert method["tcn"]["calibration_threshold_quantile"] == 0.995
    assert method["tcn"]["calibration_quantile_interpolation"] == "higher"
    assert method["tcn"]["test_tuning_allowed"] is False

    expected_points = benchmark["router"]["operating_points"]
    assert method["router"]["operating_quantiles"] == expected_points
    assert method["router"]["test_tuning_allowed"] is False

    evidence = method["evidence"]
    primary_evidence = benchmark["tcn_evidence"]
    assert evidence["tcn_score_smoothing"] == primary_evidence["score_smoothing"]
    assert evidence["tcn_persistence"] == primary_evidence["persistence"]
    assert (
        evidence["episode_merge_minutes"]
        == primary_evidence["episode_merge_minutes"]
    )
    assert (
        evidence["episode_reset_gap_minutes"]
        == primary_evidence["episode_reset_gap_minutes"]
    )


def test_transport_matches_frozen_metropt2_preprocessing() -> None:
    transport = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "energy_aware_metropt2_transport.yaml"
        ).read_text(encoding="utf-8")
    )
    report = json.loads(
        (
            ROOT
            / "docs"
            / "metropt2_preprocessing_report.json"
        ).read_text(encoding="utf-8")
    )

    assert report["feature_variants"]["flowmeter_excluded"][
        "model_feature_count"
    ] == 63

    expected = transport["data"]["expected_rows"]
    assert expected["train"] == report["splits"]["train"]["valid_bins"]
    assert (
        expected["calibration"]
        == report["splits"]["calibration"]["valid_bins"]
    )
    assert expected["test"] == report["splits"]["test"]["valid_bins"]


def test_transport_is_anchored_to_primary_result_commit() -> None:
    transport = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "energy_aware_metropt2_transport.yaml"
        ).read_text(encoding="utf-8")
    )

    assert (
        transport["transport"]["primary_result_commit"]
        == "de4dfec05167358d385c3fd21e12f195135db3fd"
    )
