from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ml.data.features import model_feature_columns
from ml.detection.alerts import (
    causal_ewma,
    extract_alert_episodes,
    persistent_alerts,
)
from ml.detection.common import fit_scaler
from ml.detection.evaluate import evaluate_detection
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.energy.hierarchical_inference import (
    routing_mask,
    sha256_file,
)
from ml.energy.primary_benchmark import (
    bin_coverage,
    episode_coverage,
    incident_related_mask,
    serializable_episodes,
    verify_frozen_metrics,
)
from ml.temporal.detector import (
    TemporalDetector,
    score_temporal_detector,
)
from ml.temporal.tcn import (
    TCNForecaster,
    TemporalForecastConfig,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG = ROOT / "configs" / "energy_aware_hierarchical_intelligence.yaml"
BENCHMARK_CONFIG = ROOT / "configs" / "energy_aware_inference_benchmark.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _load_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
    return frame.sort_index()


def _load_tcn(
    *,
    train: pd.DataFrame,
    features: list[str],
    checkpoint_path: Path,
    expected_sha: str,
    device: torch.device,
) -> tuple[TemporalDetector, float]:
    observed_sha = sha256_file(checkpoint_path)
    if observed_sha != expected_sha:
        raise RuntimeError(
            "TCN checkpoint SHA256 mismatch: "
            f"expected {expected_sha}, got {observed_sha}."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("scaler_name") != "standard":
        raise RuntimeError("Frozen TCN checkpoint must use StandardScaler.")

    checkpoint_features = [str(value) for value in checkpoint["features"]]
    if checkpoint_features != features:
        raise RuntimeError("TCN checkpoint feature order mismatch.")

    scaler = fit_scaler(
        train,
        features,
        method="standard",
    )

    location = np.asarray(checkpoint["scaler_location"], dtype=float)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=float)

    if not np.allclose(scaler.mean_, location, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Rebuilt StandardScaler mean does not match checkpoint.")
    if not np.allclose(scaler.scale_, scale, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Rebuilt StandardScaler scale does not match checkpoint.")

    model_config = checkpoint["model_config"]
    model = TCNForecaster(
        TemporalForecastConfig(
            input_dim=int(model_config["input_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            kernel_size=int(model_config["kernel_size"]),
            dilations=tuple(
                int(value)
                for value in model_config["dilations"]
            ),
            dropout=float(model_config["dropout"]),
        )
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    sequence = checkpoint["sequence"]

    detector = TemporalDetector(
        features=features,
        scaler=scaler,
        model=model,
        sequence_length=int(sequence["history_bins"]),
        bin_minutes=int(sequence["bin_minutes"]),
    )

    return detector, float(checkpoint["calibration_threshold"])


def main() -> None:
    study = _load_yaml(STUDY_CONFIG)
    benchmark = _load_yaml(BENCHMARK_CONFIG)

    primary = study["datasets"]["primary"]
    frozen_pca = study["frozen_pca"]
    tcn_cfg = benchmark["tcn"]
    evidence_cfg = benchmark["tcn_evidence"]
    outputs = benchmark["outputs"]

    router_calibration = _load_json(
        ROOT / outputs["router_calibration_file"]
    )
    preflight = _load_json(
        ROOT / outputs["pretest_preflight_file"]
    )

    if not bool(preflight.get("ready_for_primary_test")):
        raise RuntimeError("Pre-TEST preflight is not ready.")
    if preflight.get("test_file_opened") is not False:
        raise RuntimeError("Preflight provenance is invalid.")

    expected_sha = str(tcn_cfg["checkpoint_sha256"])
    if router_calibration["tcn_checkpoint"]["sha256"] != expected_sha:
        raise RuntimeError("Router calibration checkpoint SHA mismatch.")
    if preflight["tcn_checkpoint"]["sha256"] != expected_sha:
        raise RuntimeError("Preflight checkpoint SHA mismatch.")

    train = _load_partition(ROOT / primary["train_file"])
    calibration = _load_partition(ROOT / primary["calibration_file"])

    # The primary TEST is intentionally opened only after all pre-TEST
    # guards above have passed.
    test = _load_partition(ROOT / primary["test_file"])

    features = model_feature_columns(train.columns)
    if len(features) != 63:
        raise RuntimeError(
            f"Expected frozen 63-feature representation, got {len(features)}."
        )

    incidents_cfg = _load_yaml(
        ROOT / primary["incidents_config"]
    )
    incidents = incidents_cfg["dataset"]["reported_incidents"]

    # 1) Frozen operational PCA reproduction guard.
    pca = fit_pca_detector(
        train,
        features,
        variance_retained=float(frozen_pca["variance_retained"]),
        scaler_name="robust",
    )

    pca_cal_raw = score_pca_detector(calibration, pca)
    pca_cal_ewma = causal_ewma(
        pca_cal_raw,
        alpha=float(frozen_pca["alerting"]["ewma_alpha"]),
        reset_gap_minutes=int(
            frozen_pca["alerting"]["reset_gap_minutes"]
        ),
    )
    pca_alert_threshold = float(
        pca_cal_ewma.quantile(
            float(frozen_pca["alerting"]["threshold_quantile"]),
            interpolation="higher",
        )
    )

    pca_test_raw = score_pca_detector(test, pca)
    pca_test_ewma = causal_ewma(
        pca_test_raw,
        alpha=float(frozen_pca["alerting"]["ewma_alpha"]),
        reset_gap_minutes=int(
            frozen_pca["alerting"]["reset_gap_minutes"]
        ),
    )

    _, pca_alerts = persistent_alerts(
        pca_test_ewma,
        threshold=pca_alert_threshold,
        required_hits=int(
            frozen_pca["alerting"]["persistence_hits"]
        ),
        window_bins=int(
            frozen_pca["alerting"]["persistence_window"]
        ),
        reset_gap_minutes=int(
            frozen_pca["alerting"]["reset_gap_minutes"]
        ),
    )

    pca_episodes = extract_alert_episodes(
        pca_alerts,
        merge_minutes=int(
            frozen_pca["alerting"]["merge_minutes"]
        ),
        reset_gap_minutes=int(
            frozen_pca["alerting"]["reset_gap_minutes"]
        ),
    )

    evaluation_cfg = incidents_cfg["preprocessing"]
    detection_cfg = _load_yaml(ROOT / "configs" / "detection.yaml")
    detection_eval = detection_cfg["evaluation"]

    pca_metrics = evaluate_detection(
        scores=pca_test_ewma,
        alerts=pca_alerts,
        episodes=pca_episodes,
        incidents=incidents,
        early_warning_hours=int(
            detection_eval["early_warning_hours"]
        ),
        late_tolerance_hours=int(
            detection_eval["late_tolerance_hours"]
        ),
        bin_minutes=int(evaluation_cfg["bin_minutes"]),
    )

    reproduction_deltas = verify_frozen_metrics(
        pca_metrics,
        frozen_pca["expected_metrics"],
        tolerance=float(frozen_pca["reproduction_tolerance"]),
    )

    # 2) Frozen StandardScaler TCN evidence on every valid causal window.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tcn_detector, tcn_threshold = _load_tcn(
        train=train,
        features=features,
        checkpoint_path=ROOT / tcn_cfg["checkpoint_file"],
        expected_sha=expected_sha,
        device=device,
    )

    tcn_scores = score_temporal_detector(
        test,
        tcn_detector,
        device=device,
        batch_size=int(tcn_cfg["inference_batch_size"]),
    )

    if evidence_cfg["score_smoothing"] != "none":
        raise RuntimeError("TCN evidence smoothing must remain disabled.")
    if evidence_cfg["persistence"] != "none":
        raise RuntimeError("TCN evidence persistence must remain disabled.")

    tcn_alerts = (tcn_scores >= tcn_threshold).rename("alert")

    tcn_episodes = extract_alert_episodes(
        tcn_alerts,
        merge_minutes=int(
            evidence_cfg["episode_merge_minutes"]
        ),
        reset_gap_minutes=int(
            evidence_cfg["episode_reset_gap_minutes"]
        ),
    )

    related = incident_related_mask(
        pd.DatetimeIndex(tcn_scores.index),
        incidents,
        early_warning_hours=int(
            detection_eval["early_warning_hours"]
        ),
    )

    always_on_bins = bin_coverage(
        alerts=tcn_alerts,
        routed=pd.Series(
            True,
            index=tcn_alerts.index,
            dtype=bool,
        ),
        incident_related=related,
    )
    always_on_episodes = episode_coverage(
        tcn_episodes,
        alerts=tcn_alerts,
        routed=pd.Series(
            True,
            index=tcn_alerts.index,
            dtype=bool,
        ),
        incidents=incidents,
        early_warning_hours=int(
            detection_eval["early_warning_hours"]
        ),
    )

    # 3) Frozen q90/q95/q99 router coverage.
    pca_router_scores = pca_test_ewma.reindex(tcn_scores.index)

    if pca_router_scores.isna().any():
        raise RuntimeError("PCA router alignment to TCN windows contains NaN.")

    routing_results: dict[str, Any] = {}

    frozen_points = router_calibration["router"]["operating_points"]

    for operating_id in ("route_q90", "route_q95", "route_q99"):
        point = frozen_points[operating_id]
        threshold = float(point["threshold"])

        routed = routing_mask(
            pca_router_scores,
            threshold,
        )

        bin_metrics = bin_coverage(
            alerts=tcn_alerts,
            routed=routed,
            incident_related=related,
        )
        episode_metrics = episode_coverage(
            tcn_episodes,
            alerts=tcn_alerts,
            routed=routed,
            incidents=incidents,
            early_warning_hours=int(
                detection_eval["early_warning_hours"]
            ),
        )

        routing_results[operating_id] = {
            "threshold": threshold,
            "calibration_quantile": float(
                point["calibration_quantile"]
            ),
            "valid_tcn_windows": len(routed),
            "tcn_invocations": int(routed.sum()),
            "tcn_invocation_fraction": float(routed.mean()),
            **bin_metrics,
            **episode_metrics,
        }

    report = {
        "schema_version": "aeroxai.energy_aware_primary_evidence.v1",
        "study": study["study"]["name"],
        "evidence_class": "EXPLORATORY_REUSED_TEST_BENCHMARK",
        "causal_claim": False,
        "test_opened": True,
        "test_tuning_performed": False,
        "tcn_checkpoint": {
            "file": tcn_cfg["checkpoint_file"],
            "sha256": expected_sha,
        },
        "frozen_pca_reproduction": {
            "passed": True,
            "tolerance": float(
                frozen_pca["reproduction_tolerance"]
            ),
            "alert_threshold": pca_alert_threshold,
            "metrics": pca_metrics,
            "absolute_deltas": reproduction_deltas,
        },
        "always_on_tcn": {
            "valid_tcn_windows": len(tcn_scores),
            "calibration_threshold": tcn_threshold,
            "score_min": float(tcn_scores.min()),
            "score_median": float(tcn_scores.median()),
            "score_mean": float(tcn_scores.mean()),
            "score_max": float(tcn_scores.max()),
            **always_on_bins,
            **always_on_episodes,
            "episodes": serializable_episodes(tcn_episodes),
        },
        "routing": routing_results,
        "definitions": {
            "tcn_alert_bin_rule": evidence_cfg["alert_bin_rule"],
            "tcn_score_smoothing": evidence_cfg["score_smoothing"],
            "tcn_persistence": evidence_cfg["persistence"],
            "episode_coverage_rule": evidence_cfg[
                "episode_coverage_rule"
            ],
            "incident_related_window": evidence_cfg[
                "incident_related_window"
            ],
        },
    }

    output_path = ROOT / outputs["primary_evidence_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "frozen_pca_reproduction_passed": True,
                "frozen_pca_metrics": {
                    key: pca_metrics[key]
                    for key in frozen_pca["expected_metrics"]
                },
                "tcn_checkpoint_sha256": expected_sha,
                "always_on_tcn": {
                    "valid_tcn_windows": len(tcn_scores),
                    "tcn_alert_bins": always_on_bins["tcn_alert_bins"],
                    "tcn_incident_related_alert_bins": (
                        always_on_bins[
                            "tcn_incident_related_alert_bins"
                        ]
                    ),
                    "tcn_alert_episodes": (
                        always_on_episodes["alert_episodes"]
                    ),
                    "tcn_incident_related_episodes": (
                        always_on_episodes[
                            "incident_related_alert_episodes"
                        ]
                    ),
                },
                "routing": routing_results,
                "output": str(output_path.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
