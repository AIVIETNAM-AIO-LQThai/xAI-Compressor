from __future__ import annotations

from pathlib import Path

import yaml

from scripts.preflight_evidence_aware_router_v2 import (
    EXPECTED_CONVEX_WEIGHTS,
    EXPECTED_INPUTS,
    EXPECTED_SELECTION,
    EXPECTED_THRESHOLDS,
    enumerate_candidates,
    validate_preregistered_candidate_space,
)

CONFIG_PATH = Path(
    "configs/evidence_aware_router_v2.yaml"
)


def _config() -> dict:
    return yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


def test_router_v2_inputs_are_frozen() -> None:
    config = _config()

    assert config["inputs"]["ordered"] == EXPECTED_INPUTS
    assert (
        config["inputs"]["support_features_in_primary_family"]
        is False
    )
    assert config["inputs"]["tcn_score_online_input"] is False
    assert config["inputs"]["incident_label_input"] is False


def test_router_v2_candidate_grid_is_exactly_120() -> None:
    config = _config()

    candidates = validate_preregistered_candidate_space(
        config
    )

    assert len(candidates) == 120
    assert (
        len(
            [
                item
                for item in candidates
                if item["family"] == "max3"
            ]
        )
        == 15
    )
    assert (
        len(
            [
                item
                for item in candidates
                if item["family"] == "top2_mean"
            ]
        )
        == 15
    )
    assert (
        len(
            [
                item
                for item in candidates
                if item["family"] == "convex"
            ]
        )
        == 90
    )


def test_router_v2_grid_has_no_duplicates() -> None:
    config = _config()
    candidates = enumerate_candidates(config)

    identities = {
        (
            item["family"],
            item["threshold"],
            (
                None
                if item["weights"] is None
                else tuple(item["weights"])
            ),
        )
        for item in candidates
    }

    assert len(identities) == len(candidates)


def test_router_v2_thresholds_are_frozen() -> None:
    config = _config()

    assert [
        float(value)
        for value in config["thresholds"]
    ] == EXPECTED_THRESHOLDS


def test_router_v2_convex_weights_are_monotone() -> None:
    config = _config()

    weights = [
        [float(value) for value in row]
        for row in config[
            "candidate_families"
        ]["convex"]["weights"]
    ]

    assert weights == EXPECTED_CONVEX_WEIGHTS

    for row in weights:
        assert all(value >= 0.0 for value in row)
        assert abs(sum(row) - 1.0) < 1e-12


def test_router_v2_selection_order_is_frozen() -> None:
    config = _config()

    assert config["selection"] == EXPECTED_SELECTION
    assert config["no_rescue_tuning"] is True


def test_cobra_is_closed_during_v2_development() -> None:
    config = _config()

    assert all(
        value is False
        for value in config["cobra"].values()
    )
