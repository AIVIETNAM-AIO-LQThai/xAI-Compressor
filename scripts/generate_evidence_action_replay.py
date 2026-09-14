from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ml.reasoning.evidence import (
    incident_evidence_from_report,
)
from ml.reasoning.evidence_action_bridge import (
    EvidenceActionBridgeConfig,
    assess_evidence_action_bridge,
)
from ml.reasoning.hypotheses import (
    generate_candidate_hypotheses,
    load_hypothesis_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping in {path}."
        )

    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected JSON object in {path}."
        )

    return payload


def main() -> None:
    bridge_payload = _load_yaml(
        ROOT / "configs" / "evidence_action_bridge.yaml"
    )

    bridge_section = bridge_payload["bridge"]
    replay_section = bridge_payload["replay"]

    project = _load_yaml(
        ROOT / "configs" / "project.yaml"
    )

    safety = project["safety"]

    if (
        str(safety["deployment_mode"])
        != str(bridge_section["deployment_mode"])
    ):
        raise RuntimeError(
            "Bridge deployment mode does not match "
            "configs/project.yaml."
        )

    if (
        bool(safety["override_equipment_ctrl"])
        != bool(bridge_section["override_equipment_ctrl"])
    ):
        raise RuntimeError(
            "Bridge equipment-control policy does not "
            "match configs/project.yaml."
        )

    xai_report = _load_json(
        ROOT / "docs" / "xai_verification_report.json"
    )

    acceptance = xai_report.get("acceptance", {})

    if acceptance.get("all_passed") is not True:
        raise RuntimeError(
            "Verified XAI report did not pass its "
            "frozen acceptance checks."
        )

    incident_id = int(
        replay_section["incident_id"]
    )

    detector_evidence = incident_evidence_from_report(
        xai_report,
        incident_id=incident_id,
    )

    generation_config, specs = load_hypothesis_config(
        ROOT / "configs" / "waste_hypotheses.yaml"
    )

    hypotheses = generate_candidate_hypotheses(
        detector_evidence,
        specs,
        generation_config,
    )

    if not hypotheses:
        raise RuntimeError(
            "Verified detector evidence produced no "
            "candidate waste hypotheses."
        )

    bridge_config = EvidenceActionBridgeConfig(
        site_calibrated=bool(
            bridge_section["site_calibrated"]
        ),
        calibrated_compressor_inflow_available=bool(
            bridge_section[
                "calibrated_compressor_inflow_available"
            ]
        ),
        receiver_pressure_trace_available=bool(
            bridge_section[
                "receiver_pressure_trace_available"
            ]
        ),
        deployment_mode=str(
            bridge_section["deployment_mode"]
        ),
        override_equipment_ctrl=bool(
            bridge_section["override_equipment_ctrl"]
        ),
    )

    bridge = assess_evidence_action_bridge(
        bridge_config
    )

    replay = {
        "schema_version": "aeroxai.evidence_action_replay.v1",
        "incident_id": incident_id,
        "causal_claim": False,
        "detector_evidence_class": "REAL",
        "detector_evidence": asdict(detector_evidence),
        "candidate_hypotheses": [
            asdict(item)
            for item in hypotheses
        ],
        "bridge_assessment": asdict(bridge),
        "downstream_physics_executed": False,
        "robust_control_executed": False,
        "recommendation": None,
        "interpretation": (
            "AeroXAI preserved the verified real "
            "detector evidence and generated candidate "
            "waste hypotheses, then intentionally "
            "withheld real-asset physics/control "
            "reasoning because the current MetroPT "
            "prototype has no validated site-calibrated "
            "physical bridge."
        ),
    }

    if bridge.may_generate_real_asset_advisory:
        raise RuntimeError(
            "This replay is intentionally configured "
            "for the current uncalibrated MetroPT "
            "prototype. Do not silently enable the "
            "real-asset advisory path."
        )

    output_path = ROOT / replay_section["output_file"]
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            replay,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "incident_id": incident_id,
                "candidate_hypotheses": [
                    item.hypothesis_id
                    for item in hypotheses
                ],
                "bridge_status": bridge.status,
                "missing_requirements": list(
                    bridge.missing_requirements
                ),
                "downstream_physics_executed": False,
                "robust_control_executed": False,
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
