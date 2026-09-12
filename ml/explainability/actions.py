from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ActionExplanation:
    interval_index: int
    time_seconds: float
    recommended_action: str
    reason: str
    evidence_class: str
    causal_claim: bool
    demand_kg_s: float
    leak_kg_s: float
    pressure_start_bar_g: float
    pressure_end_bar_g: float
    safety_margin_bar: float
    reserve_available_kg_s: float
    required_reserve_kg_s: float
    interval_energy_kwh: float
    baseline_interval_energy_kwh: float | None
    interval_energy_delta_kwh: float | None
    binding_constraints: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _fixed_on_columns(
    schedule: pd.DataFrame,
) -> list[str]:
    return sorted(
        column
        for column in schedule.columns
        if (
            column.startswith("fixed_")
            and column.endswith("_on")
        )
    )


def _vsd_fraction_columns(
    schedule: pd.DataFrame,
) -> list[str]:
    return sorted(
        column
        for column in schedule.columns
        if column.endswith("_fraction")
    )


def _action_text(
    row: pd.Series,
    fixed_columns: list[str],
    vsd_columns: list[str],
) -> str:
    parts: list[str] = []

    for column in fixed_columns:
        compressor_id = column.removesuffix(
            "_on"
        )

        state = (
            "ON"
            if bool(row[column])
            else "OFF"
        )

        parts.append(
            f"{compressor_id} {state}"
        )

    for column in vsd_columns:
        compressor_id = (
            column.removesuffix(
                "_fraction"
            )
        )

        fraction = float(
            row[column]
        )

        if fraction <= 1.0e-10:
            parts.append(
                f"{compressor_id} OFF"
            )
        else:
            parts.append(
                
                    f"{compressor_id} "
                    f"{fraction * 100.0:.1f}%"
                
            )

    return "; ".join(parts)


def _binding_constraints(
    *,
    pressure_end_bar_g: float,
    safety_min_bar_g: float,
    safety_max_bar_g: float,
    reserve_available_kg_s: float,
    required_reserve_kg_s: float,
    tolerance: float,
) -> tuple[str, ...]:
    binding: list[str] = []

    if (
        pressure_end_bar_g
        <= safety_min_bar_g + tolerance
    ):
        binding.append(
            "pressure_safety_min"
        )

    if (
        pressure_end_bar_g
        >= safety_max_bar_g - tolerance
    ):
        binding.append(
            "pressure_safety_max"
        )

    if (
        reserve_available_kg_s
        <= required_reserve_kg_s
        + tolerance
    ):
        binding.append(
            "reserve_requirement"
        )

    return tuple(binding)


def explain_schedule(
    schedule: pd.DataFrame,
    *,
    target_bar_g: float,
    safety_min_bar_g: float,
    safety_max_bar_g: float,
    required_reserve_kg_s: float,
    baseline_interval_energy_kwh: (
        Sequence[float] | None
    ) = None,
    binding_tolerance: float = 1.0e-6,
) -> list[ActionExplanation]:
    required_columns = {
        "time_seconds",
        "demand_kg_s",
        "leak_kg_s",
        "pressure_start_bar_g",
        "pressure_bar_g",
        "reserve_available_kg_s",
        "energy_kwh",
    }

    missing = (
        required_columns
        - set(schedule.columns)
    )

    if missing:
        raise ValueError(
            "Schedule is missing required "
            f"columns: {sorted(missing)}"
        )

    if not (
        safety_min_bar_g
        < target_bar_g
        < safety_max_bar_g
    ):
        raise ValueError(
            "Pressure limits must satisfy "
            "safety_min < target < safety_max."
        )

    if required_reserve_kg_s < 0.0:
        raise ValueError(
            "required_reserve_kg_s "
            "cannot be negative."
        )

    if binding_tolerance < 0.0:
        raise ValueError(
            "binding_tolerance "
            "cannot be negative."
        )

    baseline_values: list[float] | None

    if baseline_interval_energy_kwh is None:
        baseline_values = None
    else:
        baseline_values = [
            float(value)
            for value
            in baseline_interval_energy_kwh
        ]

        if (
            len(baseline_values)
            != len(schedule)
        ):
            raise ValueError(
                "Baseline interval energy "
                "length must match schedule."
            )

    fixed_columns = _fixed_on_columns(
        schedule
    )

    vsd_columns = _vsd_fraction_columns(
        schedule
    )

    explanations: list[
        ActionExplanation
    ] = []

    for interval_index, (
        _,
        row,
    ) in enumerate(
        schedule.iterrows()
    ):
        pressure_start = float(
            row[
                "pressure_start_bar_g"
            ]
        )

        pressure_end = float(
            row[
                "pressure_bar_g"
            ]
        )

        reserve_available = float(
            row[
                "reserve_available_kg_s"
            ]
        )

        interval_energy = float(
            row["energy_kwh"]
        )

        baseline_energy = (
            None
            if baseline_values is None
            else baseline_values[
                interval_index
            ]
        )

        interval_delta = (
            None
            if baseline_energy is None
            else (
                interval_energy
                - baseline_energy
            )
        )

        binding = _binding_constraints(
            pressure_end_bar_g=(
                pressure_end
            ),
            safety_min_bar_g=(
                safety_min_bar_g
            ),
            safety_max_bar_g=(
                safety_max_bar_g
            ),
            reserve_available_kg_s=(
                reserve_available
            ),
            required_reserve_kg_s=(
                required_reserve_kg_s
            ),
            tolerance=(
                binding_tolerance
            ),
        )

        if (
            pressure_start
            > target_bar_g
        ):
            pressure_reason = (
                "Receiver pressure begins "
                "above target, so the "
                "optimized schedule can limit "
                "air production while "
                "preserving pressure and "
                "reserve constraints."
            )
        elif (
            pressure_start
            < target_bar_g
        ):
            pressure_reason = (
                "Receiver pressure begins "
                "below target, so the "
                "optimized schedule balances "
                "recovery against energy, "
                "startup, reserve, and safety "
                "constraints."
            )
        else:
            pressure_reason = (
                "Receiver pressure begins "
                "at target, so the schedule "
                "selects the lowest-objective "
                "feasible equipment dispatch "
                "for the forecast interval."
            )

        if binding:
            binding_reason = (
                " Active or near-active "
                "constraints: "
                + ", ".join(binding)
                + "."
            )
        else:
            binding_reason = (
                " No monitored safety or "
                "reserve constraint is "
                "binding at the interval end."
            )

        if (
            interval_delta is None
        ):
            comparison_reason = ""
        else:
            comparison_reason = (
                " Interval electrical energy "
                "difference versus the frozen "
                "baseline is "
                f"{interval_delta:+.6f} kWh. "
                "This is a schedule "
                "comparison, not the causal "
                "effect of one individual "
                "device action."
            )

        explanations.append(
            ActionExplanation(
                interval_index=(
                    interval_index
                ),
                time_seconds=float(
                    row["time_seconds"]
                ),
                recommended_action=(
                    _action_text(
                        row,
                        fixed_columns,
                        vsd_columns,
                    )
                ),
                reason=(
                    pressure_reason
                    + binding_reason
                    + comparison_reason
                ),
                evidence_class=(
                    "simulated"
                ),
                causal_claim=False,
                demand_kg_s=float(
                    row["demand_kg_s"]
                ),
                leak_kg_s=float(
                    row["leak_kg_s"]
                ),
                pressure_start_bar_g=(
                    pressure_start
                ),
                pressure_end_bar_g=(
                    pressure_end
                ),
                safety_margin_bar=(
                    pressure_end
                    - safety_min_bar_g
                ),
                reserve_available_kg_s=(
                    reserve_available
                ),
                required_reserve_kg_s=(
                    required_reserve_kg_s
                ),
                interval_energy_kwh=(
                    interval_energy
                ),
                baseline_interval_energy_kwh=(
                    baseline_energy
                ),
                interval_energy_delta_kwh=(
                    interval_delta
                ),
                binding_constraints=(
                    binding
                ),
            )
        )

    return explanations


def explanations_frame(
    explanations: Sequence[
        ActionExplanation
    ],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            item.to_dict()
            for item in explanations
        ]
    )
