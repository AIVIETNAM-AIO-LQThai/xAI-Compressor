from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ml.energy.rcsd1yd_evaluation import (
    ROUTE_NAMES,
    build_frozen_routes,
    build_tcn_model,
    build_temporal_batch_from_checkpoint,
    frozen_router_probability,
    score_tcn_batch,
)
from ml.energy.rcsd1yd_failure_analysis import (
    PERCENTILE_FEATURES,
    calibration_boundaries,
    descriptive_summary,
    group_summary,
    monthly_route_summary,
    probability_calibration_table,
    raw_auc_ap,
    spearman_rank_correlation,
    threshold_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rcsd1yd_router_failure_analysis.yaml"
PREREGISTRATION_COMMIT = "19aaad1611a0725501765fe4c39573e096f24c6a"


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return value


def _load_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
    return frame


def _assert_close(
    observed: float,
    expected: float,
    *,
    tolerance: float,
    label: str,
) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise RuntimeError(
            f"Closed-result reproduction failed for {label}: "
            f"{observed} != {expected}."
        )


def _partition_bundle(
    *,
    frame: pd.DataFrame,
    checkpoint: dict[str, Any],
    model,
    device: torch.device,
    adaptation: dict[str, Any],
    high_threshold: float,
    alert_threshold: float,
) -> dict[str, Any]:
    temporal_batch = build_temporal_batch_from_checkpoint(
        frame,
        checkpoint,
    )
    scores = score_tcn_batch(
        model,
        temporal_batch,
        device=device,
        batch_size=1024,
    )

    context = build_frozen_routes(
        frame=frame,
        temporal_batch=temporal_batch,
        adaptation_freeze=adaptation,
    )

    probabilities = frozen_router_probability(
        context.router_features,
        adaptation["frozen_router"],
    )

    return {
        "temporal_batch": temporal_batch,
        "scores": scores,
        "context": context,
        "probabilities": probabilities,
        "high": scores >= high_threshold,
        "alert": scores >= alert_threshold,
    }


def _teacher_summary(
    bundle: dict[str, Any],
    *,
    high_threshold: float,
    alert_threshold: float,
    score_quantiles: list[float],
) -> dict[str, Any]:
    scores = np.asarray(bundle["scores"], dtype=float)
    high = np.asarray(bundle["high"], dtype=bool)
    alert = np.asarray(bundle["alert"], dtype=bool)

    return {
        "score_distribution": descriptive_summary(
            scores,
            quantiles=score_quantiles,
        ),
        "fraction_above_frozen_cal_q90": float(high.mean()),
        "fraction_above_frozen_cal_q995": float(alert.mean()),
        "high_threshold_q90": high_threshold,
        "alert_threshold_q995": alert_threshold,
    }


def _feature_transport(
    bundle: dict[str, Any],
    *,
    quantiles: list[float],
) -> dict[str, Any]:
    context = bundle["context"]
    scores = np.asarray(bundle["scores"], dtype=float)
    high = np.asarray(bundle["high"], dtype=bool)
    result: dict[str, Any] = {}

    for feature in context.router_features.columns:
        values = context.router_features[feature].to_numpy(dtype=float)
        metrics = raw_auc_ap(values, high)

        item = {
            "distribution": descriptive_summary(
                values,
                quantiles=quantiles,
            ),
            "raw_roc_auc_vs_high_evidence": metrics["roc_auc"],
            "average_precision_vs_high_evidence": (
                metrics["average_precision"]
            ),
            "spearman_vs_raw_tcn_score": spearman_rank_correlation(
                values,
                scores,
            ),
        }

        if feature in PERCENTILE_FEATURES:
            item["fraction_exact_zero"] = float(
                np.mean(values == 0.0)
            )
            item["fraction_exact_one"] = float(
                np.mean(values == 1.0)
            )

        result[feature] = item

    return result


def _router_diagnostics(
    bundle: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    return threshold_diagnostics(
        np.asarray(bundle["probabilities"], dtype=float),
        np.asarray(bundle["high"], dtype=bool),
        threshold=threshold,
    )


def _classification_groups(bundle: dict[str, Any]) -> dict[str, Any]:
    high = np.asarray(bundle["high"], dtype=bool)
    route = np.asarray(
        bundle["context"].routes["frozen_evidence_aware_router"],
        dtype=bool,
    )

    masks = {
        "TP": route & high,
        "FN": (~route) & high,
        "FP": route & (~high),
        "TN": (~route) & (~high),
    }

    return {
        name: group_summary(
            mask=mask,
            router_probability=np.asarray(
                bundle["probabilities"],
                dtype=float,
            ),
            router_features=bundle["context"].router_features,
            pca_ewma=bundle["context"].pca_ewma.reindex(
                bundle["temporal_batch"].target_index
            ),
            raw_tcn_score=np.asarray(bundle["scores"], dtype=float),
        )
        for name, mask in masks.items()
    }


def _q90_disagreement(bundle: dict[str, Any]) -> dict[str, Any]:
    high = np.asarray(bundle["high"], dtype=bool)
    evidence = np.asarray(
        bundle["context"].routes["frozen_evidence_aware_router"],
        dtype=bool,
    )
    q90 = np.asarray(
        bundle["context"].routes["fixed_pca_route_q90"],
        dtype=bool,
    )

    masks = {
        "both": high & q90 & evidence,
        "q90_only": high & q90 & (~evidence),
        "evidence_aware_only": high & (~q90) & evidence,
        "neither": high & (~q90) & (~evidence),
    }

    high_total = int(high.sum())
    groups: dict[str, Any] = {}

    for name, mask in masks.items():
        summary = group_summary(
            mask=mask,
            router_probability=np.asarray(
                bundle["probabilities"],
                dtype=float,
            ),
            router_features=bundle["context"].router_features,
            pca_ewma=bundle["context"].pca_ewma.reindex(
                bundle["temporal_batch"].target_index
            ),
            raw_tcn_score=np.asarray(bundle["scores"], dtype=float),
        )
        summary["fraction_of_high_evidence"] = (
            summary["count"] / high_total
            if high_total
            else None
        )
        groups[name] = summary

    q90_high = high & q90
    q90_not_high = high & (~q90)

    return {
        "groups": groups,
        "conditional_evidence_aware_coverage": {
            "within_q90_routed_high_evidence": (
                float(evidence[q90_high].mean())
                if q90_high.any()
                else None
            ),
            "within_q90_not_routed_high_evidence": (
                float(evidence[q90_not_high].mean())
                if q90_not_high.any()
                else None
            ),
        },
    }


def _alert_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    alert = np.asarray(bundle["alert"], dtype=bool)
    index = pd.DatetimeIndex(bundle["temporal_batch"].target_index)
    scores = np.asarray(bundle["scores"], dtype=float)

    rows: list[dict[str, Any]] = []

    for position in np.flatnonzero(alert):
        routes = {
            name: bool(bundle["context"].routes[name][position])
            for name in ROUTE_NAMES
        }
        rows.append(
            {
                "timestamp": str(index[position]),
                "raw_tcn_score": float(scores[position]),
                "routes": routes,
            }
        )

    return rows


def _markdown(payload: dict[str, Any]) -> str:
    eval_router = payload["router_diagnostics"]["EVALUATION"]
    cal_router = payload["router_diagnostics"]["CALIBRATION"]
    fn = payload["false_negative_analysis"]["FN"]
    teacher = payload["teacher_shift"]
    q90 = payload["q90_disagreement"]

    lines = [
        "# RCSD-1YD Router Failure Analysis",
        "",
        "Evidence class: **POST_HOC_EXPLORATORY**",
        "",
        (
            "This analysis diagnoses the already-closed RCSD external transport "
            "failure. It is not independent validation and does not alter the "
            "`EXTERNAL_TRANSPORT_FAIL` classification."
        ),
        "",
        "## Closed-result reproduction",
        "",
        "All frozen closed-result guards reproduced before diagnostics ran.",
        "",
        "## Teacher shift",
        "",
        (
            f"- CAL high-evidence prevalence: "
            f"{100 * teacher['CALIBRATION']['fraction_above_frozen_cal_q90']:.3f}%"
        ),
        (
            f"- EVAL high-evidence prevalence: "
            f"{100 * teacher['EVALUATION']['fraction_above_frozen_cal_q90']:.3f}%"
        ),
        (
            f"- CAL alert prevalence: "
            f"{100 * teacher['CALIBRATION']['fraction_above_frozen_cal_q995']:.5f}%"
        ),
        (
            f"- EVAL alert prevalence: "
            f"{100 * teacher['EVALUATION']['fraction_above_frozen_cal_q995']:.5f}%"
        ),
        "",
        "## Frozen router transport",
        "",
        f"- CAL frozen-router ROC-AUC: {cal_router['roc_auc']}",
        f"- EVAL frozen-router ROC-AUC: {eval_router['roc_auc']}",
        f"- CAL average precision: {cal_router['average_precision']}",
        f"- EVAL average precision: {eval_router['average_precision']}",
        (
            f"- CAL enrichment at 0.30: "
            f"{cal_router['high_evidence_enrichment']}"
        ),
        (
            f"- EVAL enrichment at 0.30: "
            f"{eval_router['high_evidence_enrichment']}"
        ),
        "",
        "## False negatives",
        "",
        f"- EVAL high-evidence false negatives: {fn['count']}",
        "",
        "## q90 disagreement",
        "",
        (
            f"- q90-only high-evidence bins: "
            f"{q90['groups']['q90_only']['count']}"
        ),
        (
            f"- evidence-aware-only high-evidence bins: "
            f"{q90['groups']['evidence_aware_only']['count']}"
        ),
        f"- neither: {q90['groups']['neither']['count']}",
        "",
        "## Interpretation boundary",
        "",
        (
            "These diagnostics may motivate a future Router V2 hypothesis. "
            "No Router V2 policy is selected here, and any policy developed "
            "using RCSD EVALUATION requires another independent compressor "
            "dataset for primary validation."
        ),
        "",
    ]

    return "\n".join(lines)

def main() -> None:
    config = _yaml(CONFIG_PATH)

    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            PREREGISTRATION_COMMIT,
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "Failure-analysis preregistration is not an ancestor of HEAD."
        )

    outputs = config["outputs"]
    json_path = ROOT / outputs["json"]
    markdown_path = ROOT / outputs["markdown"]

    if json_path.exists() or markdown_path.exists():
        raise RuntimeError(
            "Failure-analysis output already exists; refusing to overwrite."
        )

    frozen = config["frozen_inputs"]
    external = _json(ROOT / frozen["external_result"])
    adaptation = _json(ROOT / frozen["adaptation_freeze"])

    if external["final_external_status"] != "EXTERNAL_TRANSPORT_FAIL":
        raise RuntimeError("Closed external result changed.")

    checkpoint = torch.load(
        ROOT / frozen["tcn_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    calibration = _load_partition(
        ROOT / frozen["calibration_parquet"]
    )
    evaluation = _load_partition(
        ROOT / frozen["evaluation_parquet"]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = build_tcn_model(checkpoint, device=device)

    high_threshold = float(
        adaptation["tcn"]["calibration_high_evidence_threshold_q90"]
    )
    alert_threshold = float(
        adaptation["tcn"]["calibration_alert_evidence_threshold_q995"]
    )

    cal_bundle = _partition_bundle(
        frame=calibration,
        checkpoint=checkpoint,
        model=model,
        device=device,
        adaptation=adaptation,
        high_threshold=high_threshold,
        alert_threshold=alert_threshold,
    )
    eval_bundle = _partition_bundle(
        frame=evaluation,
        checkpoint=checkpoint,
        model=model,
        device=device,
        adaptation=adaptation,
        high_threshold=high_threshold,
        alert_threshold=alert_threshold,
    )

    guard = config["reproduction_guard"]
    tolerance = float(guard["numeric_tolerance"])

    if len(eval_bundle["scores"]) != int(guard["valid_tcn_rows"]):
        raise RuntimeError("Closed valid-TCN-row count did not reproduce.")
    if int(eval_bundle["high"].sum()) != int(
        guard["evaluation_high_evidence_bins"]
    ):
        raise RuntimeError("Closed high-evidence count did not reproduce.")
    if int(eval_bundle["alert"].sum()) != int(
        guard["evaluation_alert_evidence_bins"]
    ):
        raise RuntimeError("Closed alert-evidence count did not reproduce.")

    evidence_route = eval_bundle["context"].routes[
        "frozen_evidence_aware_router"
    ]
    q90_route = eval_bundle["context"].routes[
        "fixed_pca_route_q90"
    ]

    if int(evidence_route.sum()) != int(
        guard["evidence_aware_invocations"]
    ):
        raise RuntimeError("Closed evidence-aware invocation count changed.")
    if int(q90_route.sum()) != int(guard["fixed_q90_invocations"]):
        raise RuntimeError("Closed q90 invocation count changed.")

    high = np.asarray(eval_bundle["high"], dtype=bool)
    alert = np.asarray(eval_bundle["alert"], dtype=bool)

    observed_high_coverage = float(
        (evidence_route & high).sum() / high.sum()
    )
    observed_alert_coverage = float(
        (evidence_route & alert).sum() / alert.sum()
    )
    observed_q90_high_coverage = float(
        (q90_route & high).sum() / high.sum()
    )

    _assert_close(
        observed_high_coverage,
        float(guard["evidence_aware_high_coverage"]),
        tolerance=tolerance,
        label="evidence-aware high coverage",
    )
    _assert_close(
        observed_alert_coverage,
        float(guard["evidence_aware_alert_coverage"]),
        tolerance=tolerance,
        label="evidence-aware alert coverage",
    )
    _assert_close(
        observed_q90_high_coverage,
        float(guard["fixed_q90_high_coverage"]),
        tolerance=tolerance,
        label="q90 high coverage",
    )

    if int((high & (~evidence_route)).sum()) != int(
        config["false_negative_analysis"][
            "expected_evaluation_false_negatives"
        ]
    ):
        raise RuntimeError("Closed false-negative count did not reproduce.")

    teacher_cfg = config["teacher_shift"]
    score_quantiles = [
        float(value)
        for value in teacher_cfg["score_quantiles"]
    ]

    teacher_shift = {
        "CALIBRATION": _teacher_summary(
            cal_bundle,
            high_threshold=high_threshold,
            alert_threshold=alert_threshold,
            score_quantiles=score_quantiles,
        ),
        "EVALUATION": _teacher_summary(
            eval_bundle,
            high_threshold=high_threshold,
            alert_threshold=alert_threshold,
            score_quantiles=score_quantiles,
        ),
    }

    cal_high_fraction = teacher_shift["CALIBRATION"][
        "fraction_above_frozen_cal_q90"
    ]
    eval_high_fraction = teacher_shift["EVALUATION"][
        "fraction_above_frozen_cal_q90"
    ]
    teacher_shift["eval_to_cal_high_prevalence_ratio"] = (
        eval_high_fraction / cal_high_fraction
        if cal_high_fraction > 0.0
        else None
    )

    feature_quantiles = [
        float(value)
        for value in config["cheap_feature_transport"][
            "descriptive_quantiles"
        ]
    ]

    feature_transport = {
        "CALIBRATION": _feature_transport(
            cal_bundle,
            quantiles=feature_quantiles,
        ),
        "EVALUATION": _feature_transport(
            eval_bundle,
            quantiles=feature_quantiles,
        ),
    }

    frozen_threshold = float(
        config["frozen_router_diagnostics"][
            "probability_threshold"
        ]
    )
    router_diagnostics = {
        "CALIBRATION": _router_diagnostics(
            cal_bundle,
            threshold=frozen_threshold,
        ),
        "EVALUATION": _router_diagnostics(
            eval_bundle,
            threshold=frozen_threshold,
        ),
    }

    cal_probabilities = np.asarray(
        cal_bundle["probabilities"],
        dtype=float,
    )
    eval_probabilities = np.asarray(
        eval_bundle["probabilities"],
        dtype=float,
    )

    calibration_cfg = config["frozen_router_diagnostics"][
        "calibration_bins"
    ]
    boundaries = calibration_boundaries(
        cal_probabilities,
        quantiles=[
            float(value)
            for value in calibration_cfg["quantiles"]
        ],
    )

    calibration_tables = {
        "boundaries_from_calibration": boundaries,
        "CALIBRATION": probability_calibration_table(
            cal_probabilities,
            np.asarray(cal_bundle["high"], dtype=bool),
            boundaries=boundaries,
        ),
        "EVALUATION": probability_calibration_table(
            eval_probabilities,
            high,
            boundaries=boundaries,
        ),
    }

    false_negative_analysis = _classification_groups(eval_bundle)
    q90_disagreement = _q90_disagreement(eval_bundle)

    temporal = monthly_route_summary(
        target_index=pd.DatetimeIndex(
            eval_bundle["temporal_batch"].target_index
        ),
        high_evidence=high,
        evidence_route=np.asarray(evidence_route, dtype=bool),
        q90_route=np.asarray(q90_route, dtype=bool),
        months=[
            str(value)
            for value in config["temporal_analysis"]["months"]
        ],
    )

    alert_rows = _alert_rows(eval_bundle)

    if len(alert_rows) != int(
        config["alert_analysis"]["evaluation_positive_count_expected"]
    ):
        raise RuntimeError("Alert descriptive count changed.")

    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    payload = {
        "schema_version": "aeroxai.rcsd1yd_router_failure_analysis.v1",
        "study": config["study"]["name"],
        "evidence_class": "POST_HOC_EXPLORATORY",
        "causal_claim": False,
        "closed_external_status": "EXTERNAL_TRANSPORT_FAIL",
        "closed_result_changed": False,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "implementation_commit": current_commit,
        "reproduction_guard_passed": True,
        "teacher_shift": teacher_shift,
        "cheap_feature_transport": feature_transport,
        "router_probability_distribution": {
            "CALIBRATION": descriptive_summary(
                cal_probabilities,
                quantiles=feature_quantiles,
            ),
            "EVALUATION": descriptive_summary(
                eval_probabilities,
                quantiles=feature_quantiles,
            ),
        },
        "router_diagnostics": router_diagnostics,
        "calibration_tables": calibration_tables,
        "false_negative_analysis": false_negative_analysis,
        "q90_disagreement": q90_disagreement,
        "temporal_analysis": temporal,
        "alert_analysis": {
            "positive_count": len(alert_rows),
            "descriptive_only": True,
            "rows": alert_rows,
        },
        "forbidden_operations_observed": {
            "logistic_refit": False,
            "threshold_sweep": False,
            "pca_threshold_search": False,
            "feature_selection": False,
            "new_feature_search": False,
            "tcn_retuning": False,
            "teacher_threshold_retuning": False,
            "evaluation_reference_refit": False,
            "router_v2_selection": False,
        },
        "interpretation": {
            "independent_validation": False,
            "router_v2_selected": False,
            "physical_fault_diagnosis": False,
            "external_failure_classification_remains_closed": True,
        },
    }

    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _markdown(payload),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "evidence_class": payload["evidence_class"],
                "reproduction_guard_passed": True,
                "cal_high_prevalence": teacher_shift["CALIBRATION"][
                    "fraction_above_frozen_cal_q90"
                ],
                "eval_high_prevalence": teacher_shift["EVALUATION"][
                    "fraction_above_frozen_cal_q90"
                ],
                "eval_to_cal_high_prevalence_ratio": teacher_shift[
                    "eval_to_cal_high_prevalence_ratio"
                ],
                "cal_router_auc": router_diagnostics["CALIBRATION"][
                    "roc_auc"
                ],
                "eval_router_auc": router_diagnostics["EVALUATION"][
                    "roc_auc"
                ],
                "cal_router_ap": router_diagnostics["CALIBRATION"][
                    "average_precision"
                ],
                "eval_router_ap": router_diagnostics["EVALUATION"][
                    "average_precision"
                ],
                "eval_router_enrichment": router_diagnostics["EVALUATION"][
                    "high_evidence_enrichment"
                ],
                "eval_false_negatives": false_negative_analysis["FN"][
                    "count"
                ],
                "q90_only_high_evidence": q90_disagreement["groups"][
                    "q90_only"
                ]["count"],
                "evidence_aware_only_high_evidence": (
                    q90_disagreement["groups"]["evidence_aware_only"]["count"]
                ),
                "alert_positive_count": len(alert_rows),
                "router_v2_selected": False,
                "outputs": [
                    str(json_path.relative_to(ROOT)),
                    str(markdown_path.relative_to(ROOT)),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
