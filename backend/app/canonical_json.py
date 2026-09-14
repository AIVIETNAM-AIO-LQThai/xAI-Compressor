from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            normalize_json_value(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Canonical JSON does not support "
                "NaN or infinite values."
            )

        if value.is_integer():
            return int(value)

    return value


def canonical_json_bytes(value: Any) -> bytes:
    normalized = normalize_json_value(value)

    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()
