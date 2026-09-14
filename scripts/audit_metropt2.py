from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ml.data.metropt2 import (
    audit_metropt2_csv,
    file_digests,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "metropt2.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(
            handle
        )


def main() -> None:
    config = load_config()
    dataset = config["dataset"]
    audit_config = config["audit"]

    csv_path = ROOT / dataset[
        "raw_csv"
    ]

    if not csv_path.exists():
        raise FileNotFoundError(
            "MetroPT2 CSV not found. Run:\n"
            "  python -m scripts.download_metropt2"
        )

    print(
        "Verifying MetroPT2 integrity..."
    )

    digests = file_digests(
        csv_path
    )

    expected_md5 = str(
        dataset[
            "expected_md5"
        ]
    ).lower()

    if (
        digests["md5"]
        != expected_md5
    ):
        raise RuntimeError(
            "MetroPT2 MD5 mismatch.\n"
            f"Expected: {expected_md5}\n"
            f"Actual:   {digests['md5']}"
        )

    print(
        "Scanning MetroPT2 in chunks..."
    )

    audit = audit_metropt2_csv(
        csv_path,
        chunksize=int(
            audit_config[
                "chunksize"
            ]
        ),
        reported_incidents=dataset[
            "reported_incidents"
        ],
    )

    report = {
        "schema_version": (
            "aeroxai.metropt2_data_audit.v2"
        ),
        "dataset": dataset[
            "name"
        ],
        "evidence_class": "REAL",
        "causal_claim": False,
        "source": {
            "repository": dataset[
                "source"
            ],
            "doi": dataset[
                "doi"
            ],
            "published_md5": (
                expected_md5
            ),
            "verified_md5": (
                digests["md5"]
            ),
            "local_sha256": (
                digests["sha256"]
            ),
        },
        "published_metadata_checks": {
            "expected_rows": int(
                dataset[
                    "expected_rows"
                ]
            ),
            "observed_rows": int(
                audit[
                    "row_count"
                ]
            ),
            "row_count_matches": (
                int(
                    audit[
                        "row_count"
                    ]
                )
                == int(
                    dataset[
                        "expected_rows"
                    ]
                )
            ),
            "expected_attributes": int(
                dataset[
                    "expected_attributes"
                ]
            ),
            "observed_attributes": int(
                audit[
                    "attribute_count"
                ]
            ),
            "attribute_count_matches": (
                int(
                    audit[
                        "attribute_count"
                    ]
                )
                == int(
                    dataset[
                        "expected_attributes"
                    ]
                )
            ),
        },
        "audit": audit,
        "interpretation": (
            "Data-quality audit only. Cadence is "
            "evaluated with a tolerance window because "
            "the 1 Hz logger retains fractional-second "
            "timestamps. No detector or explanation "
            "performance was evaluated in this checkpoint."
        ),
    }

    output_path = (
        ROOT
        / audit_config[
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

    cadence = audit[
        "cadence"
    ]

    print(
        json.dumps(
            {
                "dataset": report[
                    "dataset"
                ],
                "rows": audit[
                    "row_count"
                ],
                "attributes": audit[
                    "attribute_count"
                ],
                "row_count_matches": (
                    report[
                        "published_metadata_checks"
                    ][
                        "row_count_matches"
                    ]
                ),
                "attribute_count_matches": (
                    report[
                        "published_metadata_checks"
                    ][
                        "attribute_count_matches"
                    ]
                ),
                "timestamp_start": audit[
                    "timestamp_start"
                ],
                "timestamp_end": audit[
                    "timestamp_end"
                ],
                "nominal_1hz_window_seconds": (
                    cadence[
                        "nominal_window_seconds"
                    ]
                ),
                "nominal_1hz_fraction": (
                    cadence[
                        "nominal_1hz_step_fraction"
                    ]
                ),
                "short_steps_lt_0_5s": (
                    cadence[
                        "short_positive_steps_lt_0_5s"
                    ]
                ),
                "gaps_gt_1_5s": (
                    cadence[
                        "gap_steps_gt_1_5s"
                    ]
                ),
                "gaps_gt_10s": (
                    cadence[
                        "gap_steps_gt_10s"
                    ]
                ),
                "largest_gap_seconds": (
                    cadence[
                        "largest_positive_gap_seconds"
                    ]
                ),
                "observed_mean_interval_seconds": (
                    cadence[
                        "observed_mean_interval_seconds"
                    ]
                ),
                "backward_steps": audit[
                    "cadence"
                ][
                    "backward_steps"
                ],
                "reported_incidents": (
                    audit[
                        "reported_incidents"
                    ]
                ),
                "output": str(
                    output_path
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
