from __future__ import annotations

import json
from pathlib import Path

from backend.app.proof_manifest import (
    build_proof_manifest,
    write_proof_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = (
        build_proof_manifest()
    )

    output = write_proof_manifest(
        ROOT
        / "docs"
        / "proof_manifest.json"
    )

    print(
        json.dumps(
            {
                "manifest_id": (
                    manifest[
                        "manifest_id"
                    ]
                ),
                "real_bridge_status": (
                    manifest[
                        "real_evidence_path"
                    ][
                        "bridge_status"
                    ]
                ),
                "real_recommendation": (
                    manifest[
                        "real_evidence_path"
                    ][
                        "recommendation"
                    ]
                ),
                "simulation_scope": (
                    manifest[
                        "simulated_validation_path"
                    ][
                        "scope"
                    ]
                ),
                "simulation_connected_to_real_incident": (
                    manifest[
                        "simulated_validation_path"
                    ][
                        "connected_to_real_incident"
                    ]
                ),
                "simulation_robust_safe": (
                    manifest[
                        "simulated_validation_path"
                    ][
                        "robust_action"
                    ][
                        "robust_safe"
                    ]
                ),
                "output": str(
                    output
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
