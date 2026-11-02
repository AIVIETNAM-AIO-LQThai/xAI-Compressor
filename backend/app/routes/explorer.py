"""Read-only recorded-data explorer API. Timestamps are dataset local time, `YYYY-MM-DD HH:MM[:SS]`."""
import re

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.app.episode_evidence import EpisodeEvidence
from backend.app.explorer import MAX_RANGE, MAX_RAW, Explorer, IndexMissing

router = APIRouter(prefix="/explorer", tags=["explorer"])
explorer = Explorer()
evidence = EpisodeEvidence(explorer)
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")


def _time(value: str) -> pd.Timestamp:
    if not STAMP.fullmatch(value):
        raise HTTPException(400, detail={"code": "timestamp_invalid"})
    try:
        return pd.Timestamp(value.replace("T", " "))
    except ValueError as e:
        raise HTTPException(400, detail={"code": "timestamp_invalid"}) from e


def _guard(call):
    try:
        return call()
    except IndexMissing as e:
        raise HTTPException(503, detail={"code": "explorer_index_missing", "build": "python -m scripts.build_explorer_index"}) from e


@router.get("/overview")
def overview():
    return _guard(explorer.overview)


@router.get("/episodes")
def episodes():
    return _guard(lambda: {"episodes": explorer.episodes()})


@router.get("/range")
def range_(start: str, end: str):
    a, b = _time(start), _time(end)
    if b <= a or b - a > MAX_RANGE:
        raise HTTPException(400, detail={"code": "range_invalid", "max_days": MAX_RANGE.days})
    return _guard(lambda: explorer.range(a, b))


@router.get("/raw")
def raw(start: str, end: str):
    a, b = _time(start), _time(end)
    if b <= a or b - a > MAX_RAW:
        raise HTTPException(400, detail={"code": "range_invalid", "max_hours": int(MAX_RAW.total_seconds() // 3600)})
    return _guard(lambda: explorer.raw(a, b))


@router.get("/bin")
def bin_(t: str):
    stamp = _time(t)

    def get():
        try:
            return explorer.bin(stamp)
        except KeyError as e:
            raise HTTPException(404, detail={"code": "bin_not_found"}) from e

    return _guard(get)


@router.get("/episode")
def episode(start: str):
    stamp = _time(start)

    def get():
        try:
            return evidence.episode(stamp)
        except KeyError as e:
            raise HTTPException(404, detail={"code": "episode_not_found"}) from e

    return _guard(get)
