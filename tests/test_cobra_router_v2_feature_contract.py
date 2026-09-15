from __future__ import annotations

import hashlib

from scripts.derive_cobra_router_v2_feature_contract import (
    derive_feature_contract,
    exclusion_reasons,
    feature_name_sha256,
    serialize_feature_names,
)


def _rule() -> dict:
    return {
        "source": "structural_envelope_only",
        "casefold_for_rule_matching": True,
        "exclude_exact": [
            "Day",
            "Time",
            "Operating_status_PLC",
        ],
        "exclude_if_name_contains": [
            "target",
            "is_on",
            "is_open",
            "is_closed",
            "energy_consumption",
            "energy_output",
            "apparent_energy",
            "reactive_energy",
        ],
        "ordering": "lexicographic_exact_name",
    }


def test_exact_metadata_fields_are_excluded() -> None:
    fields = [
        "BPB001-[Pa_Gauge]",
        "Day",
        "Operating_status_PLC",
        "Time",
    ]

    selected, excluded = (
        derive_feature_contract(
            structural_fields=fields,
            rule=_rule(),
        )
    )

    assert selected == [
        "BPB001-[Pa_Gauge]",
    ]

    assert {
        item["name"]
        for item in excluded
    } == {
        "Day",
        "Operating_status_PLC",
        "Time",
    }


def test_contains_matching_is_casefolded() -> None:
    fields = [
        "A_current_speed",
        "B_Target_speed",
        "C_is_ON",
        "D_is_Open",
        "E_is_CLOSED",
        "F_measurement",
    ]

    selected, excluded = (
        derive_feature_contract(
            structural_fields=fields,
            rule=_rule(),
        )
    )

    assert selected == [
        "A_current_speed",
        "F_measurement",
    ]

    assert {
        item["name"]
        for item in excluded
    } == {
        "B_Target_speed",
        "C_is_ON",
        "D_is_Open",
        "E_is_CLOSED",
    }


def test_energy_counter_rules_do_not_remove_power() -> None:
    fields = [
        "A_active_energy_consumption",
        "B_energy_output",
        "C_apparent_energy",
        "D_reactive_energy",
        "E_total_effective_power",
    ]

    selected, _ = derive_feature_contract(
        structural_fields=fields,
        rule=_rule(),
    )

    assert selected == [
        "E_total_effective_power",
    ]


def test_exclusion_reasons_are_auditable() -> None:
    reasons = exclusion_reasons(
        "Pump_Target_is_ON",
        exclude_exact=set(),
        exclude_contains=(
            "target",
            "is_on",
        ),
        casefold_contains=True,
    )

    assert reasons == [
        "contains:target",
        "contains:is_on",
    ]


def test_serialization_has_final_lf() -> None:
    names = [
        "alpha",
        "beta",
    ]

    encoded = serialize_feature_names(
        names
    )

    assert encoded == b"alpha\nbeta\n"

    expected = hashlib.sha256(
        b"alpha\nbeta\n"
    ).hexdigest()

    assert (
        feature_name_sha256(names)
        == expected
    )


def test_unsorted_structural_input_is_rejected() -> None:
    fields = [
        "zeta",
        "alpha",
    ]

    try:
        derive_feature_contract(
            structural_fields=fields,
            rule=_rule(),
        )
    except RuntimeError as exc:
        assert "lexicographically ordered" in str(
            exc
        )
    else:
        raise AssertionError(
            "Expected unsorted metadata to fail."
        )
