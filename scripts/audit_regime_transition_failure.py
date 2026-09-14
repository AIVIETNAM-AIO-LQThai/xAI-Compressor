from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.alerts import (
    causal_ewma,
    extract_alert_episodes,
    persistent_alerts,
)
from ml.detection.pca_research_benchmark import (
    AlertProtocol,
)
from ml.detection.regime_pca import (
    assign_regimes,
    fit_regime_pca_detector,
    score_regime_pca_detector,
)
from ml.detection.regime_transition_ewma import (
    transition_reset_ewma,
)
from ml.detection.regime_transition_failure_diagnostics import (
    add_threshold_ratio,
    calibration_test_shift,
    episode_diagnostics,
    episode_group_summary,
    fragmentation_summary,
    occupancy_transition_summary,
    persistence_window_frame,
    relevant_episode_flags,
    transition_flags,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "regime_transition_failure_audit.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _protocol(config: dict[str, Any]) -> AlertProtocol:
    alerting = config["alerting"]
    evaluation = config["evaluation"]

    return AlertProtocol(
        ewma_alpha=float(alerting["ewma_alpha"]),
        threshold_quantile=float(
            alerting["threshold_quantile"]
        ),
        persistence_hits=int(
            alerting["persistence_hits"]
        ),
        persistence_window=int(
            alerting["persistence_window"]
        ),
        reset_gap_minutes=int(
            alerting["reset_gap_minutes"]
        ),
        merge_minutes=int(
            alerting["merge_minutes"]
        ),
        early_warning_hours=int(
            evaluation["early_warning_hours"]
        ),
        late_tolerance_hours=int(
            evaluation["late_tolerance_hours"]
        ),
        bin_minutes=int(evaluation["bin_minutes"]),
    )


def _assert_close(
    name: str,
    observed: float,
    expected: float,
    tolerance: float,
) -> None:
    if abs(observed - expected) > tolerance:
        raise RuntimeError(
            f"{name} mismatch: observed={observed!r}, "
            f"expected={expected!r}"
        )


def _episode_false_count(
    episodes: list[Any],
    incidents: list[dict[str, Any]],
    *,
    early_warning_hours: int,
) -> int:
    relevant = relevant_episode_flags(
        episodes,
        incidents,
        early_warning_hours=early_warning_hours,
    )
    return int(sum(not item for item in relevant))


def _build_pipeline(
    *,
    smoothed_calibration: pd.Series,
    smoothed_test: pd.Series,
    protocol: AlertProtocol,
) -> tuple[float, pd.Series, list[Any]]:
    threshold = float(
        smoothed_calibration.quantile(
            protocol.threshold_quantile,
            interpolation="higher",
        )
    )
    _, alerts = persistent_alerts(
        smoothed_test,
        threshold=threshold,
        required_hits=protocol.persistence_hits,
        window_bins=protocol.persistence_window,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )
    episodes = extract_alert_episodes(
        alerts,
        merge_minutes=protocol.merge_minutes,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )
    return threshold, alerts, episodes


def main() -> None:
    config = _load_yaml(CONFIG_PATH)
    dataset = config["dataset"]

    train = pd.read_parquet(
        ROOT / dataset["train_file"]
    ).sort_index()
    calibration = pd.read_parquet(
        ROOT / dataset["calibration_file"]
    ).sort_index()
    test = pd.read_parquet(
        ROOT / dataset["test_file"]
    ).sort_index()

    metropt = _load_yaml(
        ROOT / dataset["incidents_config"]
    )
    incidents = list(
        metropt["dataset"]["reported_incidents"]
    )

    features = model_feature_columns(train.columns)
    router = config["regime_router"]
    protocol = _protocol(config)
    variance_retained = float(
        config["representation"]["variance_retained"]
    )

    print("Study role:", config["study"]["role"])
    print(
        "Rows:",
        {
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
        },
    )
    print("Incidents:", len(incidents))
    print("Features:", len(features))

    detector = fit_regime_pca_detector(
        train,
        features,
        current_feature=str(router["current_feature"]),
        pressure_feature=str(router["pressure_feature"]),
        high_quantile=float(router["high_quantile"]),
        variance_retained=variance_retained,
    )

    calibration_raw = score_regime_pca_detector(
        calibration,
        detector,
    )
    test_raw = score_regime_pca_detector(
        test,
        detector,
    )
    calibration_regimes = assign_regimes(
        calibration,
        current_feature=detector.current_feature,
        pressure_feature=detector.pressure_feature,
        current_threshold=detector.current_threshold,
        pressure_threshold=detector.pressure_threshold,
    )
    test_regimes = assign_regimes(
        test,
        current_feature=detector.current_feature,
        pressure_feature=detector.pressure_feature,
        current_threshold=detector.current_threshold,
        pressure_threshold=detector.pressure_threshold,
    )

    baseline_calibration = causal_ewma(
        calibration_raw,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )
    baseline_test = causal_ewma(
        test_raw,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )
    candidate_calibration = transition_reset_ewma(
        calibration_raw,
        calibration_regimes,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )
    candidate_test = transition_reset_ewma(
        test_raw,
        test_regimes,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )

    (
        baseline_threshold,
        baseline_alerts,
        baseline_episodes,
    ) = _build_pipeline(
        smoothed_calibration=baseline_calibration,
        smoothed_test=baseline_test,
        protocol=protocol,
    )
    (
        candidate_threshold,
        candidate_alerts,
        candidate_episodes,
    ) = _build_pipeline(
        smoothed_calibration=candidate_calibration,
        smoothed_test=candidate_test,
        protocol=protocol,
    )

    reproduction = config["reproduction"]
    tolerance = float(reproduction["tolerance"])
    baseline_expected = reproduction["baseline_expected"]
    candidate_expected = reproduction["candidate_expected"]

    _assert_close(
        "baseline threshold",
        baseline_threshold,
        float(baseline_expected["threshold"]),
        tolerance,
    )
    _assert_close(
        "candidate threshold",
        candidate_threshold,
        float(candidate_expected["threshold"]),
        tolerance,
    )

    baseline_false = _episode_false_count(
        baseline_episodes,
        incidents,
        early_warning_hours=protocol.early_warning_hours,
    )
    candidate_false = _episode_false_count(
        candidate_episodes,
        incidents,
        early_warning_hours=protocol.early_warning_hours,
    )

    exact_checks = [
        (
            "baseline episodes_total",
            len(baseline_episodes),
            int(baseline_expected["episodes_total"]),
        ),
        (
            "baseline false_episodes",
            baseline_false,
            int(baseline_expected["false_episodes"]),
        ),
        (
            "candidate episodes_total",
            len(candidate_episodes),
            int(candidate_expected["episodes_total"]),
        ),
        (
            "candidate false_episodes",
            candidate_false,
            int(candidate_expected["false_episodes"]),
        ),
    ]
    for name, observed, expected in exact_checks:
        if int(observed) != int(expected):
            raise RuntimeError(
                f"{name} mismatch: {observed} != {expected}"
            )

    _assert_close(
        "baseline time_in_alert_fraction",
        float(baseline_alerts.mean()),
        float(baseline_expected["time_in_alert_fraction"]),
        tolerance,
    )
    _assert_close(
        "candidate time_in_alert_fraction",
        float(candidate_alerts.mean()),
        float(candidate_expected["time_in_alert_fraction"]),
        tolerance,
    )

    print("Baseline/candidate alert reproduction: PASS")

    shift = calibration_test_shift(
        candidate_calibration,
        calibration_regimes,
        candidate_test,
        test_regimes,
        threshold=candidate_threshold,
    )

    occupancy = {
        "calibration": occupancy_transition_summary(
            calibration_regimes,
            reset_gap_minutes=protocol.reset_gap_minutes,
        ),
        "test": occupancy_transition_summary(
            test_regimes,
            reset_gap_minutes=protocol.reset_gap_minutes,
        ),
    }

    persistence = persistence_window_frame(
        candidate_test,
        test_regimes,
        threshold=candidate_threshold,
        required_hits=protocol.persistence_hits,
        window_bins=protocol.persistence_window,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )

    transitions = transition_flags(
        test_regimes,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )

    candidate_episode_table = episode_diagnostics(
        candidate_episodes,
        persistence,
        transitions,
        incidents,
        early_warning_hours=protocol.early_warning_hours,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )
    candidate_episode_table = add_threshold_ratio(
        candidate_episode_table,
        threshold=candidate_threshold,
    )

    false_frame = candidate_episode_table.loc[
        candidate_episode_table["is_false"]
    ]
    relevant_frame = candidate_episode_table.loc[
        candidate_episode_table["is_relevant"]
    ]

    diagnostics = {
        "calibration_to_test_shift": shift,
        "occupancy_and_transitions": occupancy,
        "candidate_persistence_rows": {
            "rows": len(persistence),
            "threshold_hit_fraction": float(
                persistence["threshold_hit"].mean()
            ),
            "alert_fraction": float(
                persistence["alert"].mean()
            ),
            "alert_rows_spanning_regimes_fraction": float(
                persistence.loc[
                    persistence["alert"],
                    "history_spans_regimes",
                ].mean()
            ),
        },
        "candidate_episode_start": {
            "all": episode_group_summary(
                candidate_episode_table
            ),
            "false": episode_group_summary(
                false_frame
            ),
            "relevant": episode_group_summary(
                relevant_frame
            ),
        },
        "fragmentation": {
            "baseline": fragmentation_summary(
                baseline_episodes
            ),
            "transition_reset_candidate": (
                fragmentation_summary(
                    candidate_episodes
                )
            ),
        },
    }

    output_dir = ROOT / config["outputs"]["local_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    persistence.to_parquet(
        output_dir / "candidate_persistence_rows.parquet"
    )
    candidate_episode_table.to_parquet(
        output_dir / "candidate_episode_diagnostics.parquet",
        index=False,
    )

    payload = {
        "study": config["study"],
        "data": {
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "feature_count": len(features),
            "incident_count": len(incidents),
        },
        "reproduction": {
            "status": "PASS",
            "baseline_threshold": baseline_threshold,
            "baseline_episodes_total": len(
                baseline_episodes
            ),
            "baseline_false_episodes": baseline_false,
            "candidate_threshold": candidate_threshold,
            "candidate_episodes_total": len(
                candidate_episodes
            ),
            "candidate_false_episodes": candidate_false,
        },
        "diagnostics": diagnostics,
        "guardrails": {
            "posthoc_reused_test_diagnostic": True,
            "no_parameters_changed": True,
            "no_alternative_alert_rule_simulated": True,
        },
    }

    result_path = ROOT / config["outputs"]["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Audit complete.")
    print("Result:", result_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
