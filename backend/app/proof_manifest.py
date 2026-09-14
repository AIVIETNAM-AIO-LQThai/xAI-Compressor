from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.app.runtime import REPO_ROOT

DOCS_DIR = REPO_ROOT / "docs"


def _load_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"{path} must contain a JSON object."
        )

    return payload


def _canonical_digest(
    payload: dict[str, Any],
) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def _validate_real_path(
    real: dict[str, Any],
) -> None:
    if (
        real.get(
            "detector_evidence_class"
        )
        != "REAL"
    ):
        raise ValueError(
            "Real evidence replay must use "
            "detector_evidence_class=REAL."
        )

    if real.get("causal_claim") is not False:
        raise ValueError(
            "Real evidence replay must not "
            "contain a causal claim."
        )

    bridge = real.get(
        "bridge_assessment"
    )

    if not isinstance(
        bridge,
        dict,
    ):
        raise TypeError(
            "Real evidence replay is missing "
            "bridge_assessment."
        )

    if (
        bridge.get(
            "may_generate_real_asset_advisory"
        )
        is not False
    ):
        raise ValueError(
            "Current manifest expects the real "
            "asset advisory path to be withheld."
        )

    if (
        real.get(
            "downstream_physics_executed"
        )
        is not False
    ):
        raise ValueError(
            "Current real evidence path must "
            "stop before downstream physics."
        )

    if (
        real.get(
            "robust_control_executed"
        )
        is not False
    ):
        raise ValueError(
            "Current real evidence path must "
            "not execute robust control."
        )

    if real.get("recommendation") is not None:
        raise ValueError(
            "Current real evidence path must "
            "not contain a recommendation."
        )


def _validate_simulated_path(
    simulated: dict[str, Any],
) -> None:
    if (
        simulated.get(
            "scope"
        )
        != "SIMULATION_ONLY"
    ):
        raise ValueError(
            "Simulated proof must use "
            "scope=SIMULATION_ONLY."
        )

    if (
        simulated.get(
            "evidence_class"
        )
        != "SIMULATED"
    ):
        raise ValueError(
            "Simulated proof must use "
            "evidence_class=SIMULATED."
        )

    if (
        simulated.get(
            "connected_to_real_incident"
        )
        is not False
    ):
        raise ValueError(
            "Simulated proof must not be "
            "connected to a real incident."
        )

    if (
        simulated.get(
            "causal_claim"
        )
        is not False
    ):
        raise ValueError(
            "Simulated proof must not contain "
            "a causal claim."
        )

    recovery = simulated.get(
        "recovery_check"
    )

    if not isinstance(
        recovery,
        dict,
    ):
        raise TypeError(
            "Simulated proof is missing "
            "recovery_check."
        )

    if recovery.get("passed") is not True:
        raise ValueError(
            "Simulated inverse-physics recovery "
            "must pass before publication."
        )

    robust_action = simulated.get(
        "robust_action"
    )

    if not isinstance(
        robust_action,
        dict,
    ):
        raise TypeError(
            "Simulated proof is missing "
            "robust_action."
        )

    selected = robust_action.get(
        "selected"
    )

    if not isinstance(
        selected,
        dict,
    ):
        raise TypeError(
            "Simulated proof is missing the "
            "selected robust action."
        )

    if (
        selected.get(
            "robust_safe"
        )
        is not True
    ):
        raise ValueError(
            "Selected simulated action must "
            "pass the robust safety gate."
        )


def build_proof_manifest_from_payloads(
    *,
    real: dict[str, Any],
    simulated: dict[str, Any],
) -> dict[str, Any]:
    _validate_real_path(
        real
    )

    _validate_simulated_path(
        simulated
    )

    bridge = real[
        "bridge_assessment"
    ]

    candidate_hypotheses = [
        str(
            item[
                "hypothesis_id"
            ]
        )
        for item in real.get(
            "candidate_hypotheses",
            [],
        )
    ]

    robust_action = simulated[
        "robust_action"
    ]

    selected = robust_action[
        "selected"
    ]

    recovery = simulated[
        "recovery_check"
    ]

    state = simulated[
        "physical_state_uncertainty"
    ]

    manifest: dict[str, Any] = {
        "schema_version": (
            "aeroxai.proof_manifest.v1"
        ),
        "project": "AeroXAI",
        "causal_claim": False,
        "deployment": {
            "mode": "advisory",
            "override_equipment_ctrl": False,
        },
        "contains_evidence_classes": [
            "REAL",
            "SIMULATED",
        ],
        "real_evidence_path": {
            "source": (
                "docs/evidence_action_replay.json"
            ),
            "incident_id": real[
                "incident_id"
            ],
            "detector_evidence_class": (
                "REAL"
            ),
            "candidate_hypotheses": (
                candidate_hypotheses
            ),
            "bridge_status": bridge[
                "status"
            ],
            "physical_bridge_available": (
                bridge[
                    "may_generate_real_asset_advisory"
                ]
            ),
            "missing_requirements": list(
                bridge[
                    "missing_requirements"
                ]
            ),
            "downstream_physics_executed": (
                real[
                    "downstream_physics_executed"
                ]
            ),
            "robust_control_executed": (
                real[
                    "robust_control_executed"
                ]
            ),
            "recommendation": (
                real[
                    "recommendation"
                ]
            ),
        },
        "simulated_validation_path": {
            "source": (
                "docs/simulated_action_proof.json"
            ),
            "scope": simulated[
                "scope"
            ],
            "evidence_class": (
                "SIMULATED"
            ),
            "connected_to_real_incident": (
                simulated[
                    "connected_to_real_incident"
                ]
            ),
            "proof_id": simulated[
                "proof_id"
            ],
            "inverse_physics_recovery": {
                "passed": recovery[
                    "passed"
                ],
                "truth_total_outflow_kg_s": (
                    recovery[
                        "truth_total_outflow_kg_s"
                    ]
                ),
                "inferred_total_outflow_kg_s": (
                    recovery[
                        "inferred_total_outflow_kg_s"
                    ]
                ),
                "absolute_error_kg_s": (
                    recovery[
                        "absolute_error_kg_s"
                    ]
                ),
            },
            "physical_state": {
                "total_outflow_kg_s": (
                    state[
                        "total_outflow_kg_s"
                    ]
                ),
                "leak_identifiable": (
                    state[
                        "leak_identifiable"
                    ]
                ),
                "leak_kg_s": (
                    state[
                        "leak_kg_s"
                    ]
                ),
            },
            "robust_action": {
                "commands": (
                    selected[
                        "commands"
                    ]
                ),
                "scenario_count": (
                    robust_action[
                        "scenario_count"
                    ]
                ),
                "robust_safe": (
                    selected[
                        "robust_safe"
                    ]
                ),
                "valid_for_seconds": (
                    robust_action[
                        "valid_for_seconds"
                    ]
                ),
                "worst_case_min_pressure_bar_g": (
                    selected[
                        "worst_case_min_pressure_bar_g"
                    ]
                ),
                "worst_case_max_pressure_bar_g": (
                    selected[
                        "worst_case_max_pressure_bar_g"
                    ]
                ),
                "minimum_reserve_kg_s": (
                    selected[
                        "minimum_reserve_kg_s"
                    ]
                ),
                "horizon_energy_kwh": (
                    robust_action[
                        "robust_horizon_energy_kwh"
                    ]
                ),
                "horizon_objective_value": (
                    robust_action[
                        "robust_horizon_objective_value"
                    ]
                ),
            },
        },
        "separation_assertions": [
            (
                "The REAL MetroPT evidence path "
                "stops at the validated physical "
                "bridge boundary."
            ),
            (
                "The SIMULATED action proof is "
                "not connected to the MetroPT "
                "incident and must not be used "
                "as measured site evidence."
            ),
            (
                "No simulated leakage, energy, "
                "pressure, reserve, or compressor "
                "command is attributed to a real "
                "MetroPT incident."
            ),
        ],
        "allowed_claims": [
            (
                "Real telemetry supports anomaly "
                "detection, verified detector "
                "evidence, and candidate waste "
                "hypotheses."
            ),
            (
                "The current prototype withholds "
                "real-asset action reasoning when "
                "site calibration and required "
                "physical observations are absent."
            ),
            (
                "The synthetic digital-twin chain "
                "recovers known total outflow and "
                "produces a robust-safe simulated "
                "60-second advisory across the "
                "bounded scenario set."
            ),
        ],
        "forbidden_claims": [
            (
                "Do not claim that the synthetic "
                "physical state describes the "
                "real MetroPT incident."
            ),
            (
                "Do not claim measured MetroPT "
                "energy savings from the simulated "
                "optimizer."
            ),
            (
                "Do not claim that detector "
                "evidence establishes physical "
                "root cause."
            ),
            (
                "Do not claim that leakage is "
                "identified without independent "
                "process-demand information."
            ),
        ],
    }

    digest = _canonical_digest(
        manifest
    )

    manifest[
        "manifest_id"
    ] = (
        f"manifest_{digest[:16]}"
    )

    return manifest


def build_proof_manifest() -> dict[str, Any]:
    real = _load_json(
        DOCS_DIR
        / "evidence_action_replay.json"
    )

    simulated = _load_json(
        DOCS_DIR
        / "simulated_action_proof.json"
    )

    return build_proof_manifest_from_payloads(
        real=real,
        simulated=simulated,
    )


def write_proof_manifest(
    path: str | Path,
) -> Path:
    manifest = (
        build_proof_manifest()
    )

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path
