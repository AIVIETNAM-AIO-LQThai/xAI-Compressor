from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd

FEATURE_GROUPS: dict[str, str] = {
    "mean": "amplitude",
    "std": "amplitude",
    "rms": "amplitude",
    "mean_abs": "amplitude",
    "peak_abs": "amplitude",
    "peak_to_peak": "amplitude",
    "skewness": "distribution_shape",
    "excess_kurtosis": "distribution_shape",
    "crest_factor": "distribution_shape",
    "shape_factor": "distribution_shape",
    "impulse_factor": "distribution_shape",
    "clearance_factor": "distribution_shape",
    "zero_crossing_rate": "temporal_structure",
    "diff_rms_ratio": "temporal_structure",
    "lag1_autocorrelation": "temporal_structure",
    "spectral_centroid_norm": "spectral_shape",
    "spectral_bandwidth_norm": "spectral_shape",
    "spectral_flatness": "spectral_shape",
    "spectral_entropy": "spectral_shape",
    "spectral_rolloff_50_norm": "spectral_shape",
    "spectral_rolloff_85_norm": "spectral_shape",
    "spectral_rolloff_95_norm": "spectral_shape",
    "spectral_peak_frequency_norm": "spectral_shape",
    **{
        f"spectral_band_{index}_energy_fraction": "spectral_band_energy"
        for index in range(8)
    },
}


@dataclass(frozen=True)
class IITKRecording:
    condition: str
    reading: int
    values: np.ndarray


def archive_member(
    *,
    archive_root: str,
    condition: str,
    reading: int,
) -> str:
    return str(
        PurePosixPath(
            archive_root,
            condition,
            f"preprocess_Reading{reading}.dat",
        )
    )


def parse_recording_text(
    text: str,
) -> np.ndarray:
    normalized = (
        text.replace(",", " ")
        .replace(";", " ")
    )

    values = np.fromstring(
        normalized,
        sep=" ",
        dtype=np.float64,
    )

    if values.size == 0:
        raise ValueError(
            "Recording contains no numeric values."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Recording contains non-finite values."
        )

    return values


def read_recording(
    archive: ZipFile,
    *,
    archive_root: str,
    condition: str,
    reading: int,
) -> IITKRecording:
    member = archive_member(
        archive_root=archive_root,
        condition=condition,
        reading=reading,
    )

    text = archive.read(
        member
    ).decode(
        "utf-8",
        errors="strict",
    )

    return IITKRecording(
        condition=condition,
        reading=reading,
        values=parse_recording_text(
            text
        ),
    )


def healthy_split_for_reading(
    reading: int,
    split_config: dict,
) -> str:
    for split_name in (
        "train",
        "calibration",
        "test",
    ):
        split = split_config[
            split_name
        ]

        first = int(
            split[
                "first_reading"
            ]
        )

        last = int(
            split[
                "last_reading"
            ]
        )

        if first <= reading <= last:
            return split_name

    raise ValueError(
        "Healthy reading is outside the "
        f"pre-registered split: {reading}"
    )


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    epsilon = np.finfo(
        np.float64
    ).eps

    return float(
        numerator
        / max(
            abs(
                denominator
            ),
            epsilon,
        )
    )


def _spectral_features(
    values: np.ndarray,
    *,
    band_edges: Iterable[float],
) -> dict[str, float]:
    centered = (
        values
        - values.mean()
    )

    window = np.hanning(
        values.size
    )

    transformed = np.fft.rfft(
        centered
        * window
    )

    power = (
        np.abs(
            transformed
        )
        ** 2
    ).astype(
        np.float64,
        copy=False,
    )

    frequencies = np.linspace(
        0.0,
        1.0,
        power.size,
        dtype=np.float64,
    )

    epsilon = np.finfo(
        np.float64
    ).eps

    total_power = float(
        power.sum()
    )

    if total_power <= epsilon:
        raise ValueError(
            "Recording has effectively zero spectral power."
        )

    probabilities = (
        power
        / total_power
    )

    centroid = float(
        np.dot(
            frequencies,
            probabilities,
        )
    )

    bandwidth = float(
        np.sqrt(
            np.dot(
                (
                    frequencies
                    - centroid
                )
                ** 2,
                probabilities,
            )
        )
    )

    positive_power = (
        power
        + epsilon
    )

    flatness = float(
        np.exp(
            np.mean(
                np.log(
                    positive_power
                )
            )
        )
        / np.mean(
            positive_power
        )
    )

    entropy = float(
        -np.sum(
            probabilities
            * np.log(
                probabilities
                + epsilon
            )
        )
        / np.log(
            probabilities.size
        )
    )

    cumulative = np.cumsum(
        probabilities
    )

    def rolloff(
        fraction: float,
    ) -> float:
        index = int(
            np.searchsorted(
                cumulative,
                fraction,
                side="left",
            )
        )

        index = min(
            index,
            frequencies.size - 1,
        )

        return float(
            frequencies[
                index
            ]
        )

    peak_frequency = float(
        frequencies[
            int(
                np.argmax(
                    power
                )
            )
        ]
    )

    features = {
        "spectral_centroid_norm": (
            centroid
        ),
        "spectral_bandwidth_norm": (
            bandwidth
        ),
        "spectral_flatness": (
            flatness
        ),
        "spectral_entropy": (
            entropy
        ),
        "spectral_rolloff_50_norm": (
            rolloff(
                0.50
            )
        ),
        "spectral_rolloff_85_norm": (
            rolloff(
                0.85
            )
        ),
        "spectral_rolloff_95_norm": (
            rolloff(
                0.95
            )
        ),
        "spectral_peak_frequency_norm": (
            peak_frequency
        ),
    }

    edges = list(
        band_edges
    )

    if len(
        edges
    ) != 9:
        raise ValueError(
            "Exactly eight normalized spectral bands are required."
        )

    if (
        edges[0] != 0.0
        or edges[-1] != 1.0
    ):
        raise ValueError(
            "Spectral bands must span [0, 1]."
        )

    for index, (left, right) in enumerate(pairwise(edges)):
        if index == 7:
            mask = (
                (
                    frequencies
                    >= left
                )
                & (
                    frequencies
                    <= right
                )
            )
        else:
            mask = (
                (
                    frequencies
                    >= left
                )
                & (
                    frequencies
                    < right
                )
            )

        features[
            f"spectral_band_{index}_energy_fraction"
        ] = float(
            power[
                mask
            ].sum()
            / total_power
        )

    return features


def extract_recording_features(
    values: np.ndarray,
    *,
    spectral_band_edges: Iterable[float],
) -> dict[str, float]:
    signal = np.asarray(
        values,
        dtype=np.float64,
    )

    if signal.ndim != 1:
        raise ValueError(
            "Expected one-dimensional acoustic recording."
        )

    if signal.size < 3:
        raise ValueError(
            "Recording is too short."
        )

    if not np.isfinite(
        signal
    ).all():
        raise ValueError(
            "Recording contains non-finite values."
        )

    mean = float(
        signal.mean()
    )

    centered = (
        signal
        - mean
    )

    std = float(
        centered.std(
            ddof=0
        )
    )

    rms = float(
        np.sqrt(
            np.mean(
                signal
                ** 2
            )
        )
    )

    absolute = np.abs(
        signal
    )

    mean_abs = float(
        absolute.mean()
    )

    peak_abs = float(
        absolute.max()
    )

    peak_to_peak = float(
        np.ptp(
            signal
        )
    )

    epsilon = np.finfo(
        np.float64
    ).eps

    standardized = (
        centered
        / max(
            std,
            epsilon,
        )
    )

    skewness = float(
        np.mean(
            standardized
            ** 3
        )
    )

    excess_kurtosis = float(
        np.mean(
            standardized
            ** 4
        )
        - 3.0
    )

    root_abs_mean = float(
        np.mean(
            np.sqrt(
                absolute
            )
        )
    )

    zero_crossings = np.count_nonzero(
        np.signbit(
            centered[:-1]
        )
        != np.signbit(
            centered[1:]
        )
    )

    zero_crossing_rate = float(
        zero_crossings
        / (
            signal.size
            - 1
        )
    )

    differences = np.diff(
        signal
    )

    diff_rms = float(
        np.sqrt(
            np.mean(
                differences
                ** 2
            )
        )
    )

    x0 = centered[:-1]
    x1 = centered[1:]

    lag_denominator = float(
        np.sqrt(
            np.dot(
                x0,
                x0,
            )
            * np.dot(
                x1,
                x1,
            )
        )
    )

    lag1 = (
        0.0
        if lag_denominator
        <= epsilon
        else float(
            np.dot(
                x0,
                x1,
            )
            / lag_denominator
        )
    )

    features = {
        "mean": mean,
        "std": std,
        "rms": rms,
        "mean_abs": (
            mean_abs
        ),
        "peak_abs": (
            peak_abs
        ),
        "peak_to_peak": (
            peak_to_peak
        ),
        "skewness": (
            skewness
        ),
        "excess_kurtosis": (
            excess_kurtosis
        ),
        "crest_factor": (
            _safe_ratio(
                peak_abs,
                rms,
            )
        ),
        "shape_factor": (
            _safe_ratio(
                rms,
                mean_abs,
            )
        ),
        "impulse_factor": (
            _safe_ratio(
                peak_abs,
                mean_abs,
            )
        ),
        "clearance_factor": (
            _safe_ratio(
                peak_abs,
                root_abs_mean
                ** 2,
            )
        ),
        "zero_crossing_rate": (
            zero_crossing_rate
        ),
        "diff_rms_ratio": (
            _safe_ratio(
                diff_rms,
                rms,
            )
        ),
        "lag1_autocorrelation": (
            lag1
        ),
    }

    features.update(
        _spectral_features(
            signal,
            band_edges=(
                spectral_band_edges
            ),
        )
    )

    if set(
        features
    ) != set(
        FEATURE_GROUPS
    ):
        missing = sorted(
            set(
                FEATURE_GROUPS
            )
            - set(
                features
            )
        )

        unexpected = sorted(
            set(
                features
            )
            - set(
                FEATURE_GROUPS
            )
        )

        raise RuntimeError(
            "Feature schema mismatch. "
            f"Missing={missing}; "
            f"unexpected={unexpected}."
        )

    if not np.isfinite(
        np.asarray(
            list(
                features.values()
            ),
            dtype=np.float64,
        )
    ).all():
        raise RuntimeError(
            "Extracted non-finite features."
        )

    return features


def feature_group_columns(
    columns: Iterable[str],
) -> dict[str, list[str]]:
    output: dict[
        str,
        list[str],
    ] = {}

    for column in columns:
        if column not in (
            FEATURE_GROUPS
        ):
            continue

        group = FEATURE_GROUPS[
            column
        ]

        output.setdefault(
            group,
            [],
        ).append(
            column
        )

    return output


def model_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in (
            FEATURE_GROUPS
        )
        if column in frame.columns
    ]
