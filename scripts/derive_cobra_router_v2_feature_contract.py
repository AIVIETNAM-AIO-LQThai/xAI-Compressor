from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT / "configs/cobra_router_v2_adaptation.yaml"
)

STRUCTURAL_CONTRACT_PATH = (
    ROOT
    / "docs/research/cobra_structural_compatibility_contract.json"
)

OUTPUT_PATH = (
    ROOT
    / "docs/research/cobra_router_v2_raw_feature_contract.json"
)

PREREGISTRATION_COMMIT = (
    "a8f0072806a4ce9ea275e03d584c4ee363084164"
)

IMPLEMENTATION_PATHS = [
    "scripts/derive_cobra_router_v2_feature_contract.py",
    "tests/test_cobra_router_v2_feature_contract.py",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.resolve() != CONFIG_PATH.resolve():
        raise RuntimeError(
            f"Unauthorized YAML input path: {path}"
        )

    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected YAML mapping in {path}."
        )

    return payload


def _load_json(path: Path) -> dict[str, Any]:
    if path.resolve() != STRUCTURAL_CONTRACT_PATH.resolve():
        raise RuntimeError(
            f"Unauthorized JSON input path: {path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected JSON mapping in {path}."
        )

    return payload


def serialize_feature_names(
    names: list[str],
) -> bytes:
    return (
        "".join(f"{name}\n" for name in names)
        .encode("utf-8")
    )


def feature_name_sha256(
    names: list[str],
) -> str:
    return hashlib.sha256(
        serialize_feature_names(names)
    ).hexdigest()


def exclusion_reasons(
    name: str,
    *,
    exclude_exact: set[str],
    exclude_contains: tuple[str, ...],
    casefold_contains: bool,
) -> list[str]:
    reasons: list[str] = []

    if name in exclude_exact:
        reasons.append(f"exact:{name}")

    haystack = (
        name.casefold()
        if casefold_contains
        else name
    )

    for token in exclude_contains:
        needle = (
            token.casefold()
            if casefold_contains
            else token
        )

        if needle in haystack:
            reasons.append(
                f"contains:{token}"
            )

    return reasons


def derive_feature_contract(
    *,
    structural_fields: list[str],
    rule: dict[str, Any],
) -> tuple[
    list[str],
    list[dict[str, Any]],
]:
    if len(structural_fields) != len(
        set(structural_fields)
    ):
        raise RuntimeError(
            "Structural envelope contains duplicate names."
        )

    if structural_fields != sorted(
        structural_fields
    ):
        raise RuntimeError(
            "Structural envelope must be lexicographically ordered."
        )

    if str(rule["source"]) != (
        "structural_envelope_only"
    ):
        raise RuntimeError(
            "Feature derivation source changed."
        )

    if str(rule["ordering"]) != (
        "lexicographic_exact_name"
    ):
        raise RuntimeError(
            "Final feature ordering changed."
        )

    exclude_exact = {
        str(value)
        for value in rule["exclude_exact"]
    }

    exclude_contains = tuple(
        str(value)
        for value in rule[
            "exclude_if_name_contains"
        ]
    )

    casefold_contains = bool(
        rule["casefold_for_rule_matching"]
    )

    selected: list[str] = []
    excluded: list[dict[str, Any]] = []

    for name in structural_fields:
        reasons = exclusion_reasons(
            name,
            exclude_exact=exclude_exact,
            exclude_contains=exclude_contains,
            casefold_contains=casefold_contains,
        )

        if reasons:
            excluded.append(
                {
                    "name": name,
                    "reasons": reasons,
                }
            )
        else:
            selected.append(name)

    selected = sorted(selected)

    if not selected:
        raise RuntimeError(
            "Metadata rule produced an empty feature set."
        )

    if len(selected) + len(excluded) != len(
        structural_fields
    ):
        raise RuntimeError(
            "Feature derivation cardinality mismatch."
        )

    return selected, excluded


def _verify_execution_boundary() -> None:
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
            "CoBra adaptation preregistration is "
            "not an ancestor of HEAD."
        )

    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            *IMPLEMENTATION_PATHS,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    if status.stdout.strip():
        raise RuntimeError(
            "Feature-contract implementation is "
            "not clean and committed."
        )

    if OUTPUT_PATH.exists():
        raise RuntimeError(
            "Feature-contract output already exists; "
            "refusing to overwrite."
        )


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def main() -> None:
    _verify_execution_boundary()

    config = _load_yaml(CONFIG_PATH)
    structural = _load_json(
        STRUCTURAL_CONTRACT_PATH
    )

    study = config["study"]

    if study["status"] != (
        "preregistered_before_cobra_sensor_values"
    ):
        raise RuntimeError(
            "Unexpected CoBra adaptation status."
        )

    if study[
        "parent_router_v2_policy_commit"
    ] != "392c54f4bae7be632c89ef677b09aa44c09dfec5":
        raise RuntimeError(
            "Frozen Router V2 policy linkage changed."
        )

    source_cfg = config["schema"][
        "structural_envelope_source"
    ]

    if structural["schema_version"] != (
        "aeroxai.cobra_structural_compatibility.v1"
    ):
        raise RuntimeError(
            "Unexpected structural contract schema."
        )

    structural_fields = [
        str(value)
        for value in structural[
            "candidate_structural_fields"
        ]
    ]

    expected_count = int(
        source_cfg["expected_count"]
    )

    if len(structural_fields) != expected_count:
        raise RuntimeError(
            "Structural feature count changed."
        )

    if int(
        structural[
            "candidate_structural_field_count"
        ]
    ) != expected_count:
        raise RuntimeError(
            "Structural contract count mismatch."
        )

    structural_sha = feature_name_sha256(
        structural_fields
    )

    expected_sha = str(
        source_cfg["expected_sha256"]
    )

    if structural_sha != expected_sha:
        raise RuntimeError(
            "Structural envelope SHA256 differs "
            "from preregistration."
        )

    if structural[
        "candidate_feature_sha256"
    ] != expected_sha:
        raise RuntimeError(
            "Committed structural SHA256 differs "
            "from adaptation config."
        )

    interpretation = structural[
        "interpretation"
    ]

    for key in (
        "cobra_sensor_values_used",
        "cobra_model_outcomes_used",
        "evaluation_behavior_used",
    ):
        if bool(interpretation[key]):
            raise RuntimeError(
                f"Structural firewall failed: {key}"
            )

    rule = config[
        "final_raw_feature_rule"
    ]

    selected, excluded = (
        derive_feature_contract(
            structural_fields=structural_fields,
            rule=rule,
        )
    )

    final_sha = feature_name_sha256(
        selected
    )

    payload = {
        "schema_version": (
            "aeroxai.cobra_router_v2_raw_feature_contract.v1"
        ),
        "study": "cobra_router_v2_adaptation",
        "status": (
            "FROZEN_METADATA_ONLY_RAW_FEATURE_SET"
        ),
        "preregistration_commit": (
            PREREGISTRATION_COMMIT
        ),
        "derivation_implementation_commit": (
            _git_head()
        ),
        "router_v2_policy_commit": (
            "392c54f4bae7be632c89ef677b09aa44c09dfec5"
        ),
        "source": {
            "path": str(
                STRUCTURAL_CONTRACT_PATH.relative_to(
                    ROOT
                )
            ),
            "structural_field_count": (
                len(structural_fields)
            ),
            "structural_field_sha256": (
                structural_sha
            ),
        },
        "derivation_rule": {
            "source": rule["source"],
            "casefold_for_rule_matching": bool(
                rule[
                    "casefold_for_rule_matching"
                ]
            ),
            "exclude_exact": list(
                rule["exclude_exact"]
            ),
            "exclude_if_name_contains": list(
                rule[
                    "exclude_if_name_contains"
                ]
            ),
            "ordering": rule["ordering"],
        },
        "excluded_field_count": len(excluded),
        "excluded_fields": excluded,
        "final_raw_feature_count": len(
            selected
        ),
        "final_raw_feature_sha256": (
            final_sha
        ),
        "final_raw_features": selected,
        "serialization": (
            "UTF-8 exact feature names, "
            "lexicographic order, LF-separated, "
            "final LF"
        ),
        "source_access": {
            "files_read": [
                str(
                    CONFIG_PATH.relative_to(ROOT)
                ),
                str(
                    STRUCTURAL_CONTRACT_PATH.relative_to(
                        ROOT
                    )
                ),
            ],
            "cobra_archives_opened": False,
            "cobra_csv_files_opened": False,
            "cobra_sensor_values_used": False,
            "cobra_sensor_distributions_used": False,
            "cobra_model_outcomes_used": False,
            "cobra_evaluation_behavior_used": False,
        },
        "is_final_model_raw_feature_set": True,
        "data_driven_feature_selection": False,
        "evaluation_specific_feature_selection": False,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "structural_field_count": (
                    len(structural_fields)
                ),
                "excluded_field_count": (
                    len(excluded)
                ),
                "final_raw_feature_count": (
                    len(selected)
                ),
                "final_raw_feature_sha256": (
                    final_sha
                ),
                "cobra_sensor_values_used": (
                    False
                ),
                "cobra_model_outcomes_used": (
                    False
                ),
                "output": str(
                    OUTPUT_PATH.relative_to(
                        ROOT
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
