from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.explainability.evidence_states import (
    build_evidence_state_frame,
    classify_magnitude_state,
    classify_spread_state,
    fit_scalar_reference,
    scalar_percentiles,
    state_runs,
)


def test_scalar_percentiles_use_tie_aware_midranks():
    reference = fit_scalar_reference(
        pd.Series(
            [
                1.0,
                2.0,
                2.0,
                4.0,
            ]
        )
    )

    percentiles = scalar_percentiles(
        pd.Series(
            [
                2.0,
            ]
        ),
        reference,
    )

    # left rank = 1, right rank = 3,
    # midrank = 2, divided by n=4.
    assert (
        percentiles.iloc[0]
        == pytest.approx(0.5)
    )


def test_spread_state_uses_calibration_percentiles():
    percentiles = pd.Series(
        [
            0.05,
            0.50,
            0.95,
            np.nan,
        ]
    )

    states = classify_spread_state(
        percentiles,
        concentrated_percentile_max=0.10,
        diffuse_percentile_min=0.90,
    )

    assert states.tolist() == [
        "concentrated",
        "typical",
        "diffuse",
        "unavailable",
    ]


def test_alerting_takes_precedence_over_current_score():
    index = pd.date_range(
        "2022-06-01",
        periods=3,
        freq="5min",
    )

    score = pd.Series(
        [
            0.5,
            1.5,
            0.8,
        ],
        index=index,
    )

    alerts = pd.Series(
        [
            False,
            False,
            True,
        ],
        index=index,
    )

    states = classify_magnitude_state(
        score=score,
        threshold=1.0,
        alerts=alerts,
    )

    assert states.tolist() == [
        "subthreshold",
        "threshold_crossing",
        "alerting",
    ]


def test_build_state_frame_combines_axes():
    index = pd.date_range(
        "2022-06-01",
        periods=3,
        freq="5min",
    )

    trajectory = pd.DataFrame(
        {
            "effective_group_count": [
                1.0,
                2.0,
                3.0,
            ],
            "top1_concentration": [
                0.9,
                0.6,
                0.4,
            ],
            "top3_concentration": [
                1.0,
                1.0,
                1.0,
            ],
            "dominant_group": [
                "a",
                "b",
                "c",
            ],
        },
        index=index,
    )

    score = pd.Series(
        [
            0.5,
            1.5,
            2.0,
        ],
        index=index,
    )

    alerts = pd.Series(
        [
            False,
            False,
            True,
        ],
        index=index,
    )

    score_reference = fit_scalar_reference(
        pd.Series(
            [
                0.1,
                0.5,
                1.0,
                1.5,
            ]
        )
    )

    spread_reference = fit_scalar_reference(
        pd.Series(
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ]
        )
    )

    result = build_evidence_state_frame(
        trajectory=trajectory,
        score=score,
        alerts=alerts,
        threshold=1.0,
        score_reference=score_reference,
        spread_reference=spread_reference,
        concentrated_percentile_max=0.30,
        diffuse_percentile_min=0.60,
    )

    assert result[
        "magnitude_state"
    ].tolist() == [
        "subthreshold",
        "threshold_crossing",
        "alerting",
    ]

    assert result[
        "spread_state"
    ].tolist() == [
        "concentrated",
        "typical",
        "diffuse",
    ]

    assert result[
        "joint_state"
    ].tolist() == [
        "subthreshold|concentrated",
        "threshold_crossing|typical",
        "alerting|diffuse",
    ]


def test_state_runs_preserve_transition_order():
    index = pd.date_range(
        "2022-06-01",
        periods=5,
        freq="5min",
    )

    frame = pd.DataFrame(
        {
            "joint_state": [
                "subthreshold|diffuse",
                "subthreshold|diffuse",
                "threshold_crossing|diffuse",
                "alerting|typical",
                "alerting|typical",
            ]
        },
        index=index,
    )

    runs = state_runs(
        frame
    )

    assert [
        item["state"]
        for item in runs
    ] == [
        "subthreshold|diffuse",
        "threshold_crossing|diffuse",
        "alerting|typical",
    ]

    assert [
        item["bins"]
        for item in runs
    ] == [
        2,
        1,
        2,
    ]
