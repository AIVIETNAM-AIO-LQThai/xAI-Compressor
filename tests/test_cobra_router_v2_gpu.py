from __future__ import annotations

from scripts.benchmark_cobra_router_v2_gpu import (
    energy_status,
    frozen_condition_orders,
    overall_status,
)


def test_condition_orders_are_frozen() -> None:
    assert frozen_condition_orders(
        5
    ) == [
        [
            "router_v2",
            "always_on_tcn",
        ],
        [
            "router_v2",
            "always_on_tcn",
        ],
        [
            "router_v2",
            "always_on_tcn",
        ],
        [
            "always_on_tcn",
            "router_v2",
        ],
        [
            "router_v2",
            "always_on_tcn",
        ],
    ]


def test_energy_success_requires_mean_and_direction() -> None:
    status, mean, positive = energy_status(
        [
            0.60,
            0.55,
            0.52,
            0.51,
            0.50,
        ],
        minimum_mean=0.50,
        positive_each_required=True,
    )

    assert status == "PASS"
    assert mean >= 0.50
    assert positive is True


def test_energy_failure_below_mean_target() -> None:
    status, _, _ = energy_status(
        [
            0.20,
            0.25,
            0.22,
            0.24,
            0.23,
        ],
        minimum_mean=0.50,
        positive_each_required=True,
    )

    assert status == "FAIL"


def test_overall_alert_insufficient_is_not_full_pass() -> None:
    assert overall_status(
        high_status="PASS",
        alert_status="INSUFFICIENT_EVIDENCE",
        gpu_energy_status="PASS",
    ) == (
        "PARTIAL_PASS_ALERT_"
        "INSUFFICIENT_EVIDENCE"
    )


def test_energy_failure_makes_overall_fail() -> None:
    assert overall_status(
        high_status="PASS",
        alert_status="INSUFFICIENT_EVIDENCE",
        gpu_energy_status="FAIL",
    ) == "FAIL"


def test_score_loader_accepts_unnamed_timestamp_index(
    tmp_path,
) -> None:
    import pandas as pd

    from scripts.benchmark_cobra_router_v2_gpu import (
        load_frozen_score_file,
    )

    path = tmp_path / "scores.csv"

    source = pd.DataFrame(
        {
            "route_tcn": [True, False],
            "tcn_score": [0.1, 0.2],
        },
        index=pd.to_datetime(
            [
                "2025-02-17 10:00:00",
                "2025-02-17 10:05:00",
            ]
        ),
    )

    # Deliberately leave index.name = None,
    # matching the frozen CSV serialization case.
    source.to_csv(path)

    loaded = load_frozen_score_file(path)

    assert loaded.index.name == "timestamp"
    assert isinstance(
        loaded.index,
        pd.DatetimeIndex,
    )
    assert loaded["route_tcn"].tolist() == [
        True,
        False,
    ]
