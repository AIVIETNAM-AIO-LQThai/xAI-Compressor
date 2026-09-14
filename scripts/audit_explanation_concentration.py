from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ml.explainability.concentration import (
    explanation_concentration,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "metropt2.yaml"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(
            f"{path} must contain a JSON object."
        )

    return payload


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    config = _load_config()
    concentration_config = config["concentration"]
    source_path = ROOT / concentration_config["source_report"]
    source = _load_json(source_path)

    acceptance = source.get("acceptance", {})

    if acceptance.get("all_passed") is not True:
        raise RuntimeError(
            "Frozen XAI verification report did not pass "
            "all acceptance checks."
        )

    incident_rows = []

    for incident in source["incidents"]:
        shares = {
            str(item["group"]): float(item["share"])
            for item in incident["ranked_groups"]
        }
        metrics = explanation_concentration(shares)
        ranked = sorted(
            shares.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        incident_rows.append(
            {
                "id": int(incident["id"]),
                "timing": incident["timing"],
                "explanation_timestamp": incident[
                    "explanation_timestamp"
                ],
                "top_groups": [
                    {"group": group, "share": share}
                    for group, share in ranked[:5]
                ],
                "concentration": asdict(metrics),
            }
        )

    effective_counts = [
        item["concentration"]["effective_group_count"]
        for item in incident_rows
    ]
    top3_values = [
        item["concentration"]["top3_concentration"]
        for item in incident_rows
    ]

    report = {
        "schema_version": "aeroxai.explanation_concentration.v1",
        "dataset": source["dataset"],
        "evidence_class": "REAL",
        "causal_claim": False,
        "source_report": str(
            concentration_config["source_report"]
        ),
        "detector": source["detector"],
        "metric_definitions": {
            "top_k_concentration": (
                "Sum of the k largest normalized "
                "operator-level contribution shares."
            ),
            "shannon_entropy_nats": (
                "-sum(p_i * ln(p_i)) over positive group "
                "contribution shares."
            ),
            "effective_group_count": (
                "exp(Shannon entropy); diversity-equivalent "
                "number of equally contributing groups."
            ),
            "herfindahl_index": (
                "sum(p_i^2); larger values indicate more "
                "concentrated explanations."
            ),
        },
        "incidents": incident_rows,
        "summary": {
            "incident_count": len(incident_rows),
            "mean_top3_concentration": statistics.fmean(
                top3_values
            ),
            "min_top3_concentration": min(top3_values),
            "max_top3_concentration": max(top3_values),
            "mean_effective_group_count": statistics.fmean(
                effective_counts
            ),
            "min_effective_group_count": min(
                effective_counts
            ),
            "max_effective_group_count": max(
                effective_counts
            ),
        },
        "interpretation": (
            "Concentration describes how detector evidence "
            "is distributed across operator-level groups. "
            "It does not measure physical fault complexity "
            "or causal mechanism count."
        ),
    }

    output_path = ROOT / concentration_config["report_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "incidents": [
                    {
                        "id": item["id"],
                        "dominant_group": item["concentration"][
                            "dominant_group"
                        ],
                        "top1": item["concentration"][
                            "top1_concentration"
                        ],
                        "top3": item["concentration"][
                            "top3_concentration"
                        ],
                        "effective_groups": item[
                            "concentration"
                        ]["effective_group_count"],
                    }
                    for item in incident_rows
                ],
                "summary": report["summary"],
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
