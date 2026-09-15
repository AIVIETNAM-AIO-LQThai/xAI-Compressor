from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

V2_FEATURE_NAMES = [
    "pca_ewma_percentile",
    "recent_q90_hit_fraction",
    "recent_q95_hit_fraction",
]

FAMILY_ORDER = {
    "max3": 0,
    "top2_mean": 1,
    "convex": 2,
}


@dataclass(frozen=True)
class RouterV2Candidate:
    family: str
    threshold: float
    weights: tuple[float, float, float] | None = None


def candidate_grid(
    config: dict[str, Any],
) -> list[RouterV2Candidate]:
    thresholds = [
        float(value)
        for value in config["thresholds"]
    ]
    families = config["candidate_families"]

    candidates: list[RouterV2Candidate] = []

    if bool(families["max3"]["enabled"]):
        candidates.extend(
            RouterV2Candidate(
                family="max3",
                threshold=threshold,
            )
            for threshold in thresholds
        )

    if bool(families["top2_mean"]["enabled"]):
        candidates.extend(
            RouterV2Candidate(
                family="top2_mean",
                threshold=threshold,
            )
            for threshold in thresholds
        )

    if bool(families["convex"]["enabled"]):
        for row in families["convex"]["weights"]:
            weights = tuple(
                float(value)
                for value in row
            )

            if len(weights) != 3:
                raise ValueError(
                    "Router V2 convex weights must "
                    "contain exactly three values."
                )

            if any(value < 0.0 for value in weights):
                raise ValueError(
                    "Router V2 convex weights must "
                    "be nonnegative."
                )

            if abs(sum(weights) - 1.0) > 1.0e-12:
                raise ValueError(
                    "Router V2 convex weights must "
                    "sum to one."
                )

            for threshold in thresholds:
                candidates.append(
                    RouterV2Candidate(
                        family="convex",
                        threshold=threshold,
                        weights=weights,
                    )
                )

    if len(candidates) != int(
        config["expected_candidate_count"]
    ):
        raise RuntimeError(
            "Router V2 candidate count differs "
            "from preregistration."
        )

    identities = [
        (
            candidate.family,
            candidate.threshold,
            candidate.weights,
        )
        for candidate in candidates
    ]

    if len(identities) != len(set(identities)):
        raise RuntimeError(
            "Router V2 candidate grid contains duplicates."
        )

    return candidates


def _feature_matrix(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if frame.columns.tolist() != V2_FEATURE_NAMES:
        raise ValueError(
            "Router V2 feature identity/order changed."
        )

    values = frame.to_numpy(dtype=float)

    finite = np.isfinite(values)

    finite_values = values[finite]
    if (
        np.any(finite_values < 0.0)
        or np.any(finite_values > 1.0)
    ):
        raise ValueError(
            "Finite Router V2 inputs must lie in [0, 1]."
        )

    valid_rows = finite.all(axis=1)

    return values, valid_rows


def candidate_scores(
    frame: pd.DataFrame,
    candidate: RouterV2Candidate,
) -> np.ndarray:
    values, valid_rows = _feature_matrix(frame)

    scores = np.full(
        len(frame),
        np.nan,
        dtype=float,
    )

    x = values[valid_rows]

    if candidate.family == "max3":
        scores[valid_rows] = np.max(
            x,
            axis=1,
        )

    elif candidate.family == "top2_mean":
        ordered = np.sort(
            x,
            axis=1,
        )
        scores[valid_rows] = np.mean(
            ordered[:, -2:],
            axis=1,
        )

    elif candidate.family == "convex":
        if candidate.weights is None:
            raise ValueError(
                "Convex Router V2 candidate "
                "requires weights."
            )

        weights = np.asarray(
            candidate.weights,
            dtype=float,
        )

        if weights.shape != (3,):
            raise ValueError(
                "Convex Router V2 weight shape changed."
            )

        if np.any(weights < 0.0):
            raise ValueError(
                "Convex Router V2 weights are negative."
            )

        if abs(float(weights.sum()) - 1.0) > 1.0e-12:
            raise ValueError(
                "Convex Router V2 weights do not sum "
                "to one."
            )

        scores[valid_rows] = x @ weights

    else:
        raise ValueError(
            f"Unknown Router V2 family: "
            f"{candidate.family}"
        )

    return scores


def route_candidate(
    frame: pd.DataFrame,
    candidate: RouterV2Candidate,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not 0.0 <= candidate.threshold <= 1.0:
        raise ValueError(
            "Router V2 threshold must lie in [0, 1]."
        )

    _, valid_rows = _feature_matrix(frame)

    scores = candidate_scores(
        frame,
        candidate,
    )

    route = np.zeros(
        len(frame),
        dtype=bool,
    )

    route[valid_rows] = (
        scores[valid_rows]
        >= candidate.threshold
    )

    # Frozen conservative fallback:
    # valid TCN target + unavailable router input -> TCN.
    route[~valid_rows] = True

    return (
        route,
        scores,
        int((~valid_rows).sum()),
    )


def evidence_coverage(
    route: np.ndarray,
    evidence: np.ndarray,
) -> float | None:
    route = np.asarray(
        route,
        dtype=bool,
    )
    evidence = np.asarray(
        evidence,
        dtype=bool,
    )

    if route.shape != evidence.shape:
        raise ValueError(
            "Route/evidence shapes differ."
        )

    denominator = int(evidence.sum())

    if denominator == 0:
        return None

    return float(
        (route & evidence).sum()
        / denominator
    )


def serialize_candidate(
    candidate: RouterV2Candidate,
) -> dict[str, Any]:
    return {
        "family": candidate.family,
        "threshold": candidate.threshold,
        "weights": (
            None
            if candidate.weights is None
            else list(candidate.weights)
        ),
    }


def evaluate_candidate(
    *,
    candidate: RouterV2Candidate,
    domains: dict[str, dict[str, Any]],
    minimum_high_units: int,
    minimum_alert_units: int,
    minimum_high_coverage: float,
    minimum_alert_coverage: float,
) -> dict[str, Any]:
    domain_reports: dict[str, Any] = {}

    invocation = []
    applied_high = []
    applied_alert = []

    eligible = True

    for name, domain in domains.items():
        features = domain["features"]

        high = np.asarray(
            domain["high"],
            dtype=bool,
        )
        alert = np.asarray(
            domain["alert"],
            dtype=bool,
        )

        if not (
            len(features)
            == high.size
            == alert.size
        ):
            raise ValueError(
                f"{name} Router V2 arrays are "
                "misaligned."
            )

        route, _, forced_missing = route_candidate(
            features,
            candidate,
        )

        high_count = int(high.sum())
        alert_count = int(alert.sum())

        high_coverage = evidence_coverage(
            route,
            high,
        )
        alert_coverage = evidence_coverage(
            route,
            alert,
        )

        high_constraint_applied = (
            high_count >= minimum_high_units
        )
        alert_constraint_applied = (
            alert_count >= minimum_alert_units
        )

        high_pass = (
            not high_constraint_applied
            or (
                high_coverage is not None
                and high_coverage
                >= minimum_high_coverage
            )
        )

        alert_pass = (
            not alert_constraint_applied
            or (
                alert_coverage is not None
                and alert_coverage
                >= minimum_alert_coverage
            )
        )

        eligible = (
            eligible
            and high_pass
            and alert_pass
        )

        invocation_fraction = float(
            route.mean()
        )

        invocation.append(
            invocation_fraction
        )

        if high_constraint_applied:
            if high_coverage is None:
                raise RuntimeError(
                    "Applied high-evidence constraint "
                    "has no coverage value."
                )
            applied_high.append(
                float(high_coverage)
            )

        if alert_constraint_applied:
            if alert_coverage is None:
                raise RuntimeError(
                    "Applied alert constraint has no "
                    "coverage value."
                )
            applied_alert.append(
                float(alert_coverage)
            )

        domain_reports[name] = {
            "rows": len(features),
            "forced_missing_input_routes": (
                forced_missing
            ),
            "tcn_invocation_fraction": (
                invocation_fraction
            ),
            "high_evidence_units": high_count,
            "high_evidence_coverage": (
                high_coverage
            ),
            "high_constraint_applied": (
                high_constraint_applied
            ),
            "high_constraint_pass": high_pass,
            "alert_evidence_units": alert_count,
            "alert_evidence_coverage": (
                alert_coverage
            ),
            "alert_constraint_applied": (
                alert_constraint_applied
            ),
            "alert_constraint_pass": alert_pass,
        }

    if not applied_high:
        eligible = False

    return {
        **serialize_candidate(candidate),
        "domains": domain_reports,
        "worst_domain_tcn_invocation_fraction": (
            max(invocation)
        ),
        "mean_domain_tcn_invocation_fraction": (
            float(np.mean(invocation))
        ),
        "worst_domain_high_evidence_coverage": (
            min(applied_high)
            if applied_high
            else None
        ),
        "worst_applicable_alert_coverage": (
            min(applied_alert)
            if applied_alert
            else None
        ),
        "eligible": eligible,
    }


def select_candidate(
    reports: list[dict[str, Any]],
    *,
    convex_weight_order: list[
        tuple[float, float, float]
    ],
) -> dict[str, Any] | None:
    eligible = [
        report
        for report in reports
        if bool(report["eligible"])
    ]

    if not eligible:
        return None

    def selection_key(
        report: dict[str, Any],
    ) -> tuple[
        float,
        float,
        float,
        float,
        int,
        float,
        int,
    ]:
        worst_high = report[
            "worst_domain_high_evidence_coverage"
        ]

        if worst_high is None:
            raise RuntimeError(
                "Eligible candidate has no "
                "high-evidence coverage."
            )

        alert = report[
            "worst_applicable_alert_coverage"
        ]
        alert_value = (
            0.0
            if alert is None
            else float(alert)
        )

        family = str(report["family"])

        if family not in FAMILY_ORDER:
            raise RuntimeError(
                "Unknown family during Router V2 "
                "selection."
            )

        convex_index = -1

        if family == "convex":
            weights = tuple(
                float(value)
                for value in report["weights"]
            )
            convex_index = (
                convex_weight_order.index(weights)
            )

        return (
            float(
                report[
                    "worst_domain_tcn_invocation_fraction"
                ]
            ),
            float(
                report[
                    "mean_domain_tcn_invocation_fraction"
                ]
            ),
            -float(worst_high),
            -alert_value,
            FAMILY_ORDER[family],
            -float(report["threshold"]),
            convex_index,
        )

    return min(
        eligible,
        key=selection_key,
    )
