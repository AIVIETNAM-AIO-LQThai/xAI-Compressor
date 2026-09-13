from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ml.reasoning.evidence import (
    IncidentEvidence,
)


@dataclass(frozen=True)
class HypothesisSpec:
    id: str
    label: str
    description: str
    trigger_groups: tuple[str, ...]
    required_observations: tuple[str, ...]
    latent_variables: tuple[str, ...]
    alternatives: tuple[str, ...]
    physics_verifiable: bool


@dataclass(frozen=True)
class CandidateHypothesis:
    hypothesis_id: str
    label: str
    matched_groups: tuple[str, ...]
    strongest_share: float
    strongest_percentile: float
    counterfactual_support_groups: tuple[
        str,
        ...
    ]
    evidence_class: str
    causal_claim: bool


@dataclass(frozen=True)
class CandidateGenerationConfig:
    top_groups: int
    minimum_share: float
    minimum_percentile: float


def _parse_hypothesis(
    item: dict[str, Any],
) -> HypothesisSpec:
    return HypothesisSpec(
        id=str(item["id"]),
        label=str(item["label"]),
        description=str(
            item["description"]
        ),
        trigger_groups=tuple(
            str(value)
            for value in item[
                "trigger_groups"
            ]
        ),
        required_observations=tuple(
            str(value)
            for value in item[
                "required_observations"
            ]
        ),
        latent_variables=tuple(
            str(value)
            for value in item[
                "latent_variables"
            ]
        ),
        alternatives=tuple(
            str(value)
            for value in item.get(
                "alternatives",
                [],
            )
        ),
        physics_verifiable=bool(
            item[
                "physics_verifiable"
            ]
        ),
    )


def load_hypothesis_config(
    path: str | Path,
) -> tuple[
    CandidateGenerationConfig,
    tuple[HypothesisSpec, ...],
]:
    config_path = Path(path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(handle)

    generation = payload[
        "candidate_generation"
    ]

    generation_config = (
        CandidateGenerationConfig(
            top_groups=int(
                generation["top_groups"]
            ),
            minimum_share=float(
                generation[
                    "minimum_share"
                ]
            ),
            minimum_percentile=float(
                generation[
                    "minimum_percentile"
                ]
            ),
        )
    )

    specs = tuple(
        _parse_hypothesis(item)
        for item in payload[
            "hypotheses"
        ]
    )

    ids = [
        spec.id
        for spec in specs
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Hypothesis ids must be unique."
        )

    known_ids = set(ids)

    for spec in specs:
        unknown_alternatives = (
            set(spec.alternatives)
            - known_ids
        )

        if unknown_alternatives:
            raise ValueError(
                "Unknown alternative hypothesis "
                f"for {spec.id}: "
                f"{sorted(unknown_alternatives)}"
            )

    return (
        generation_config,
        specs,
    )


def generate_candidate_hypotheses(
    evidence: IncidentEvidence,
    specs: tuple[
        HypothesisSpec,
        ...
    ],
    config: CandidateGenerationConfig,
) -> tuple[
    CandidateHypothesis,
    ...
]:
    if config.top_groups <= 0:
        raise ValueError(
            "top_groups must be positive."
        )

    selected_groups = tuple(
        group
        for group in evidence.groups[
            : config.top_groups
        ]
        if (
            group.share
            >= config.minimum_share
            and group.calibration_percentile
            >= config.minimum_percentile
        )
    )

    candidates: list[
        CandidateHypothesis
    ] = []

    for spec in specs:
        matched = tuple(
            group
            for group in selected_groups
            if group.group
            in spec.trigger_groups
        )

        if not matched:
            continue

        strongest = max(
            matched,
            key=lambda item: (
                item.share
            ),
        )

        counterfactual_groups = tuple(
            item.group
            for item in matched
            if (
                item
                .counterfactual_alert_cleared
                is True
            )
        )

        candidates.append(
            CandidateHypothesis(
                hypothesis_id=spec.id,
                label=spec.label,
                matched_groups=tuple(
                    item.group
                    for item in matched
                ),
                strongest_share=float(
                    strongest.share
                ),
                strongest_percentile=float(
                    strongest
                    .calibration_percentile
                ),
                counterfactual_support_groups=(
                    counterfactual_groups
                ),
                evidence_class="MODEL_INFERENCE_FROM_REAL",
                causal_claim=False,
            )
        )

    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.strongest_share,
                item.strongest_percentile,
            ),
            reverse=True,
        )
    )