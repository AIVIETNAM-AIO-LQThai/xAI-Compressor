from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import yaml

from ml.data.iitk_air_compressor import (
    archive_member,
    parse_recording_text,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "iitk_air_compressor.yaml"
)


def _load_config() -> dict:
    return yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:
    config = _load_config()

    dataset = config[
        "dataset"
    ]

    zip_path = ROOT / dataset[
        "raw_zip"
    ]

    expected_count = int(
        dataset[
            "readings_per_condition"
        ]
    )

    expected_samples = int(
        dataset[
            "samples_per_recording"
        ]
    )

    condition_map = {
        "healthy": dataset[
            "conditions"
        ][
            "healthy"
        ],
        **dataset[
            "conditions"
        ][
            "faults"
        ],
    }

    report = {
        "schema_version": (
            "aeroxai.iitk_air_compressor_data_audit.v1"
        ),
        "dataset": (
            dataset[
                "short_name"
            ]
        ),
        "zip_sha256": (
            _sha256(
                zip_path
            )
        ),
        "archive_root": (
            dataset[
                "archive_root"
            ]
        ),
        "expected_readings_per_condition": (
            expected_count
        ),
        "expected_samples_per_recording": (
            expected_samples
        ),
        "conditions": {},
        "all_conditions_complete": True,
        "all_recordings_numeric_finite": True,
        "all_recordings_expected_length": True,
        "total_recordings": 0,
        "sample_rate_hz": (
            dataset[
                "sample_rate_hz"
            ]
        ),
        "sample_rate_status": (
            dataset[
                "sample_rate_status"
            ]
        ),
        "absolute_frequency_features_allowed": (
            dataset[
                "absolute_frequency_features"
            ]
        ),
    }

    with ZipFile(
        zip_path
    ) as archive:
        names = set(
            archive.namelist()
        )

        for label, condition in (
            condition_map.items()
        ):
            lengths = []
            missing = []
            invalid = []
            record_means = []
            record_stds = []
            record_mins = []
            record_maxs = []

            for reading in range(
                1,
                expected_count
                + 1,
            ):
                member = (
                    archive_member(
                        archive_root=dataset[
                            "archive_root"
                        ],
                        condition=condition,
                        reading=reading,
                    )
                )

                if member not in names:
                    missing.append(
                        reading
                    )
                    continue

                text = archive.read(
                    member
                ).decode(
                    "utf-8",
                    errors="strict",
                )

                try:
                    values = (
                        parse_recording_text(
                            text
                        )
                    )
                except ValueError:
                    invalid.append(
                        reading
                    )
                    continue

                lengths.append(
                    int(
                        values.size
                    )
                )

                record_means.append(
                    float(
                        values.mean()
                    )
                )

                record_stds.append(
                    float(
                        values.std(
                            ddof=0
                        )
                    )
                )

                record_mins.append(
                    float(
                        values.min()
                    )
                )

                record_maxs.append(
                    float(
                        values.max()
                    )
                )

            length_counts = {
                str(
                    length
                ): count
                for length, count
                in sorted(
                    Counter(
                        lengths
                    ).items()
                )
            }

            complete = (
                not missing
                and len(
                    lengths
                )
                == expected_count
            )

            finite = (
                not invalid
            )

            expected_length = (
                bool(
                    lengths
                )
                and all(
                    length
                    == expected_samples
                    for length in lengths
                )
            )

            report[
                "conditions"
            ][label] = {
                "archive_directory": (
                    condition
                ),
                "recording_count": (
                    len(
                        lengths
                    )
                ),
                "missing_readings": (
                    missing
                ),
                "invalid_readings": (
                    invalid
                ),
                "sample_count_distribution": (
                    length_counts
                ),
                "recording_mean_range": (
                    None
                    if not record_means
                    else [
                        min(
                            record_means
                        ),
                        max(
                            record_means
                        ),
                    ]
                ),
                "recording_std_range": (
                    None
                    if not record_stds
                    else [
                        min(
                            record_stds
                        ),
                        max(
                            record_stds
                        ),
                    ]
                ),
                "signal_min": (
                    None
                    if not record_mins
                    else min(
                        record_mins
                    )
                ),
                "signal_max": (
                    None
                    if not record_maxs
                    else max(
                        record_maxs
                    )
                ),
            }

            report[
                "all_conditions_complete"
            ] &= complete

            report[
                "all_recordings_numeric_finite"
            ] &= finite

            report[
                "all_recordings_expected_length"
            ] &= expected_length

            report[
                "total_recordings"
            ] += len(
                lengths
            )

    report[
        "balanced_condition_counts"
    ] = (
        len(
            {
                condition[
                    "recording_count"
                ]
                for condition
                in report[
                    "conditions"
                ].values()
            }
        )
        == 1
    )

    report[
        "audit_passed"
    ] = all(
        [
            report[
                "all_conditions_complete"
            ],
            report[
                "all_recordings_numeric_finite"
            ],
            report[
                "all_recordings_expected_length"
            ],
            report[
                "balanced_condition_counts"
            ],
            report[
                "total_recordings"
            ]
            == (
                expected_count
                * len(
                    condition_map
                )
            ),
        ]
    )

    output_path = (
        ROOT
        / config[
            "audit"
        ][
            "report_file"
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    if not report[
        "audit_passed"
    ]:
        raise SystemExit(
            "IITK dataset audit failed."
        )


if __name__ == "__main__":
    main()
