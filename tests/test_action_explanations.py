import pandas as pd
import pytest

from ml.explainability.actions import (
    explain_schedule,
    explanations_frame,
)


def example_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_seconds": [
                60.0,
                120.0,
            ],
            "demand_kg_s": [
                0.07,
                0.11,
            ],
            "leak_kg_s": [
                0.005,
                0.005,
            ],
            "pressure_start_bar_g": [
                7.0,
                6.6,
            ],
            "pressure_bar_g": [
                6.9,
                6.5,
            ],
            "reserve_available_kg_s": [
                0.025,
                0.010,
            ],
            "energy_kwh": [
                0.30,
                0.40,
            ],
            "fixed_1_on": [
                False,
                True,
            ],
            "fixed_2_on": [
                False,
                False,
            ],
            "vsd_1_fraction": [
                0.75,
                0.70,
            ],
        }
    )


def test_action_text_and_noncausal_flag():
    explanations = explain_schedule(
        example_schedule(),
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        required_reserve_kg_s=0.01,
    )

    first = explanations[0]

    assert (
        "fixed_1 OFF"
        in first.recommended_action
    )

    assert (
        "vsd_1 75.0%"
        in first.recommended_action
    )

    assert not first.causal_claim

    assert (
        first.evidence_class
        == "simulated"
    )


def test_binding_constraints_are_reported():
    explanations = explain_schedule(
        example_schedule(),
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        required_reserve_kg_s=0.01,
    )

    second = explanations[1]

    assert (
        "pressure_safety_min"
        in second.binding_constraints
    )

    assert (
        "reserve_requirement"
        in second.binding_constraints
    )


def test_baseline_interval_delta():
    explanations = explain_schedule(
        example_schedule(),
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        required_reserve_kg_s=0.01,
        baseline_interval_energy_kwh=[
            0.35,
            0.45,
        ],
    )

    assert (
        explanations[0]
        .interval_energy_delta_kwh
        == pytest.approx(-0.05)
    )


def test_baseline_length_must_match():
    with pytest.raises(
        ValueError,
        match="length must match",
    ):
        explain_schedule(
            example_schedule(),
            target_bar_g=7.0,
            safety_min_bar_g=6.5,
            safety_max_bar_g=7.5,
            required_reserve_kg_s=0.01,
            baseline_interval_energy_kwh=[
                0.35
            ],
        )


def test_frame_conversion():
    explanations = explain_schedule(
        example_schedule(),
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        required_reserve_kg_s=0.01,
    )

    frame = explanations_frame(
        explanations
    )

    assert len(frame) == 2
    assert "reason" in frame.columns
