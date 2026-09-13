from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ml.twin.compressor import CompressorSpec


@dataclass(frozen=True)
class CompressorRuntimeState:
    compressor_id: str
    is_on: bool
    seconds_in_state: float

    def __post_init__(self) -> None:
        if not self.compressor_id:
            raise ValueError(
                "compressor_id cannot be empty."
            )

        if (
            not math.isfinite(
                self.seconds_in_state
            )
            or self.seconds_in_state < 0.0
        ):
            raise ValueError(
                "seconds_in_state must be a "
                "finite non-negative value."
            )


def resolve_runtime_state(
    compressors: Sequence[CompressorSpec],
    states: Sequence[
        CompressorRuntimeState
    ] | None,
) -> tuple[CompressorRuntimeState, ...]:
    if states is None:
        return tuple(
            CompressorRuntimeState(
                compressor_id=spec.id,
                is_on=False,
                seconds_in_state=(
                    spec.min_off_seconds
                ),
            )
            for spec in compressors
        )

    state_tuple = tuple(states)

    state_by_id = {
        state.compressor_id: state
        for state in state_tuple
    }

    if len(state_by_id) != len(
        state_tuple
    ):
        raise ValueError(
            "Runtime-state compressor ids "
            "must be unique."
        )

    expected_ids = {
        spec.id
        for spec in compressors
    }

    if set(state_by_id) != expected_ids:
        raise ValueError(
            "Runtime-state compressor ids "
            "must exactly match the fleet."
        )

    return tuple(
        state_by_id[spec.id]
        for spec in compressors
    )


def advance_runtime_state(
    previous_states: Sequence[
        CompressorRuntimeState
    ],
    commands: (
        Mapping[str, float]
        | Sequence[tuple[str, float]]
    ),
    *,
    interval_seconds: float,
    compressors: Sequence[CompressorSpec],
) -> tuple[CompressorRuntimeState, ...]:
    if interval_seconds <= 0.0:
        raise ValueError(
            "interval_seconds must be positive."
        )

    previous = resolve_runtime_state(
        compressors,
        previous_states,
    )

    command_map = dict(commands)

    expected_ids = {
        spec.id
        for spec in compressors
    }

    if set(command_map) != expected_ids:
        raise ValueError(
            "Command compressor ids must "
            "exactly match the fleet."
        )

    previous_by_id = {
        state.compressor_id: state
        for state in previous
    }

    next_states: list[
        CompressorRuntimeState
    ] = []

    for spec in compressors:
        command = float(
            command_map[spec.id]
        )

        if (
            not math.isfinite(command)
            or command < 0.0
        ):
            raise ValueError(
                "Compressor commands must be "
                "finite non-negative values."
            )

        next_is_on = command > 0.0
        prior = previous_by_id[
            spec.id
        ]

        if next_is_on == prior.is_on:
            seconds_in_state = (
                prior.seconds_in_state
                + interval_seconds
            )
        else:
            seconds_in_state = (
                interval_seconds
            )

        next_states.append(
            CompressorRuntimeState(
                compressor_id=spec.id,
                is_on=next_is_on,
                seconds_in_state=(
                    seconds_in_state
                ),
            )
        )

    return tuple(next_states)
