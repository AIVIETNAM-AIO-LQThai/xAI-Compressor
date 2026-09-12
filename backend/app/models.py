from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

EvidenceClass = Literal[
    "REAL",
    "SIMULATED",
    "LITERATURE",
]


class TwinSimulationRequest(BaseModel):
    initial_pressure_bar_g: float = Field(
        default=7.0,
        ge=0.0,
    )
    mass_flow_in_kg_s: float = Field(
        default=0.10,
        ge=0.0,
    )
    demand_mass_flow_kg_s: float = Field(
        default=0.09,
        ge=0.0,
    )
    leak_mass_flow_kg_s: float = Field(
        default=0.005,
        ge=0.0,
    )
    duration_seconds: int = Field(
        default=600,
        ge=1,
        le=3600,
    )
    timestep_seconds: int = Field(
        default=1,
        ge=1,
        le=60,
    )

    @model_validator(mode="after")
    def validate_duration(self) -> TwinSimulationRequest:
        if (
            self.duration_seconds
            % self.timestep_seconds
            != 0
        ):
            raise ValueError(
                "duration_seconds must be "
                "divisible by timestep_seconds."
            )

        return self


class TwinTrajectoryPoint(BaseModel):
    time_seconds: float
    pressure_bar_g: float


class TwinSimulationResponse(BaseModel):
    evidence_class: EvidenceClass
    initial_pressure_bar_g: float
    final_pressure_bar_g: float
    minimum_pressure_bar_g: float
    maximum_pressure_bar_g: float
    duration_seconds: int
    timestep_seconds: int
    trajectory: list[
        TwinTrajectoryPoint
    ]
    limitations: list[str]


class OptimizationRecommendRequest(BaseModel):
    initial_pressure_bar_g: float = Field(
        default=7.0,
        ge=0.0,
    )
    demand_profile_kg_s: list[float] = Field(
        min_length=1,
        max_length=60,
    )
    leak_mass_flow_kg_s: float = Field(
        default=0.005,
        ge=0.0,
    )

    @model_validator(mode="after")
    def validate_demand(
        self,
    ) -> OptimizationRecommendRequest:
        if any(
            value < 0.0
            for value in self.demand_profile_kg_s
        ):
            raise ValueError(
                "Demand values cannot be negative."
            )

        return self


class CompressorAction(BaseModel):
    compressor_id: str
    kind: Literal[
        "fixed",
        "vsd",
    ]
    state: Literal[
        "on",
        "off",
    ]
    load_fraction: float


class RecommendationExplanation(BaseModel):
    reason: str
    predicted_pressure_end_bar_g: float
    safety_margin_bar: float
    reserve_available_kg_s: float
    required_reserve_kg_s: float
    binding_constraints: list[str]
    causal_claim: bool = False


class RecommendationSafety(BaseModel):
    deployment_mode: Literal[
        "advisory"
    ]
    override_equipment_ctrl: Literal[
        False
    ]
    valid_for_seconds: float
    requires_reoptimization: Literal[
        True
    ]
    open_loop_schedule_approved: bool
    robustness_status: str


class OptimizationRecommendResponse(BaseModel):
    evidence_class: EvidenceClass
    solver_status: int
    solver_message: str
    horizon_intervals: int
    predicted_horizon_energy_kwh: float
    current_actions: list[
        CompressorAction
    ]
    explanation: RecommendationExplanation
    safety: RecommendationSafety
    limitations: list[str]


class DetectionEvidenceResponse(BaseModel):
    evidence_class: EvidenceClass
    dataset: str
    detector: dict[str, Any]
    metrics: dict[str, Any]
    incidents: list[dict[str, Any]]
    explanation_basis: str
    causal_claim: bool
    limitations: list[str]


class EvidenceSummaryResponse(BaseModel):
    project: str
    deployment: dict[str, Any]
    evidence_classes: dict[
        str,
        Any,
    ]
    safety: dict[str, Any]
