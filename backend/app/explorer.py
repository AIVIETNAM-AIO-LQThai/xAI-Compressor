"""Recorded-data explorer: an index of the whole MetroPT-3 recording and read-only queries over it.

The index is rebuilt from local files by `python -m scripts.build_explorer_index` and written to data/processed/explorer/:
- raw/<day>.parquet   the 15 recorded channels at 10 s, exactly as normalised by ml/data/schema.py;
- detector.parquet    every valid five-minute bin with the frozen PCA detector's raw and smoothed score;
- episodes.json       the test-period alert episodes with their link to a documented incident under the evaluation's own rule;
- manifest.json       source checksums, the frozen values the build reproduced, and checksums of every output.

Nothing is interpolated: a bin missing from the frozen features (coverage below 80%) is a gap, and queries return a gap row there so a
chart breaks its line. The detector is refit deterministically from the frozen training split; the build stops unless it reproduces the
frozen threshold, the frozen Incident 2 smoothed score and the frozen episode counts. Alerts exist only in the test period: the training
period is in-sample and the calibration period sets the threshold, so neither has alert decisions.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.data.schema import (
    ANALOG_COLUMNS,
    DIGITAL_COLUMNS,
    SENSOR_COLUMNS,
    normalize_raw_frame,
)
from ml.detection.alerts import causal_ewma, extract_alert_episodes, persistent_alerts
from ml.detection.pca_detector import fit_pca_detector, score_pca_detector

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "data" / "processed" / "explorer"
METRO, DETECTION = ROOT / "configs" / "metropt.yaml", ROOT / "configs" / "detection.yaml"
FROZEN_XAI, FROZEN_BENCH = ROOT / "docs" / "xai_verification_report.json", ROOT / "docs" / "detector_benchmark.json"
BIN = pd.Timedelta(minutes=5)
RAW_STEP = pd.Timedelta(seconds=10)
MAX_RANGE = pd.Timedelta(days=14)  # five-minute detail
MAX_RAW = pd.Timedelta(hours=24)  # ten-second detail
STATS = ("mean", "min", "max")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cfg():
    return yaml.safe_load(METRO.read_text("utf-8")), yaml.safe_load(DETECTION.read_text("utf-8"))


def _linked(episode, incidents: list[dict], early_hours: int) -> int | None:
    """The evaluation's rule for an episode that counts toward an incident (ml/detection/evaluate.py, relevant episodes)."""
    for i in incidents:
        start, end = pd.Timestamp(i["start"]), pd.Timestamp(i["end"])
        if episode.start <= end and episode.end >= start - pd.Timedelta(hours=early_hours):
            return i["id"]
    return None


def build_index(out: Path = INDEX) -> dict:
    """Rebuild the index from the local raw recording and frozen splits; refuse to write it unless the frozen values are reproduced."""
    metro, det = _cfg()
    ds, pre, al = metro["dataset"], metro["preprocessing"], det["alerting"]
    splits = {name: pd.read_parquet(ROOT / pre[f"{name}_file"]) for name in ("train", "calibration", "test")}
    features = model_feature_columns(splits["train"].columns)
    detector = fit_pca_detector(splits["train"], features, variance_retained=det["pca"]["variance_retained"], scaler_name=det["pca"]["scaler"])

    parts = []
    for name, frame in splits.items():
        raw = score_pca_detector(frame, detector)
        smoothed = causal_ewma(raw, alpha=al["ewma_alpha"], reset_gap_minutes=al["reset_gap_minutes"])
        gap = raw.index.to_series().diff() > pd.Timedelta(minutes=al["reset_gap_minutes"])
        gap.iloc[0] = True
        parts.append(pd.DataFrame({"split": name, "raw_score": raw, "smoothed_score": smoothed, "restart": gap.to_numpy(),
                                   "coverage_ratio": frame["coverage_ratio"], "sample_count": frame["sample_count"]}))
    series = pd.concat(parts).sort_index()
    cal = series[series.split == "calibration"]["smoothed_score"]
    threshold = float(cal.quantile(al["threshold_quantile"], interpolation="higher"))
    series["threshold_hit"] = series["smoothed_score"] >= threshold
    test = series[series.split == "test"]["smoothed_score"]
    _, alerts = persistent_alerts(test, threshold=threshold, required_hits=al["persistence_hits"], window_bins=al["persistence_window"],
                                  reset_gap_minutes=al["reset_gap_minutes"])
    series["alert"] = pd.Series(pd.NA, index=series.index, dtype="boolean")
    series.loc[alerts.index, "alert"] = alerts.astype("boolean")
    episodes = extract_alert_episodes(alerts, merge_minutes=al["merge_minutes"], reset_gap_minutes=al["reset_gap_minutes"])
    early = det["evaluation"]["early_warning_hours"]
    rows = []
    for e in episodes:
        window = series.loc[e.start:e.end, "smoothed_score"]
        rows.append({"start": str(e.start), "end": str(e.end), "alert_bins": e.alert_bins, "peak_smoothed": float(window.max()),
                     "peak_time": str(window.idxmax()), "linked_incident": _linked(e, ds["reported_incidents"], early)})

    # The build is only valid if it reproduces what was frozen.
    xai = json.loads(FROZEN_XAI.read_text("utf-8"))
    bench = json.loads(FROZEN_BENCH.read_text("utf-8"))["details"]["pca_robust"]
    inc2 = next(i for i in xai["incidents"] if i["id"] == 2)
    at = pd.Timestamp(inc2["explanation_timestamp"])
    checks = {
        "threshold": [threshold, inc2["threshold"]],
        "incident_2_smoothed": [float(series.loc[at, "smoothed_score"]), inc2["original_smoothed_score"]],
        "episodes_total": [len(rows), bench["episodes_total"]],
        "episodes_not_linked": [sum(r["linked_incident"] is None for r in rows), bench["false_episodes"]],
    }
    failed = [k for k, (got, want) in checks.items() if not np.isclose(got, want, rtol=1e-9, atol=1e-9)]
    if failed:
        raise RuntimeError(f"explorer index does not reproduce the frozen values: {failed}")

    raw = normalize_raw_frame(pd.read_csv(ROOT / ds["raw_csv"]))
    if len(raw) != ds["expected_rows"]:
        raise RuntimeError(f"raw recording has {len(raw)} rows, expected {ds['expected_rows']}")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    for day, frame in raw.groupby(raw["timestamp"].dt.date):
        frame.astype({c: "float32" for c in ANALOG_COLUMNS} | {c: "int8" for c in DIGITAL_COLUMNS}).to_parquet(out / "raw" / f"{day}.parquet", index=False)
    series.to_parquet(out / "detector.parquet")
    (out / "episodes.json").write_text(json.dumps(rows, indent=1), "utf-8")

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    sources = [ROOT / ds["raw_zip"], ROOT / ds["raw_csv"], *(ROOT / pre[f"{n}_file"] for n in ("train", "calibration", "test")),
               ROOT / pre["features_file"], METRO, DETECTION, FROZEN_XAI, FROZEN_BENCH]
    manifest = {
        "schema": "aeroxai.explorer.v1",
        "code_commit": head,
        "sources": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sources},
        "raw_zip_matches_config": sha256(ROOT / ds["raw_zip"]) == ds["expected_zip_sha256"],
        "reproduced": checks,
        "threshold": threshold,
        "components": int(detector.model.n_components_),
        "bins": {name: int((series.split == name).sum()) for name in ("train", "calibration", "test")},
        "raw_rows": len(raw),
        "outputs": {str(p.relative_to(out)).replace("\\", "/"): sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
    return manifest


class IndexMissing(Exception):
    pass


def _iso(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def _num(values, digits: int) -> list:
    """JSON numbers with gaps as null."""
    return [None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), digits) for v in values]


def _with_gaps(frame: pd.DataFrame, step: pd.Timedelta) -> pd.DataFrame:
    """Insert one empty row after every gap longer than the sampling step, so a chart breaks its line instead of joining across it."""
    if frame.empty:
        return frame
    times = frame.index.to_series()
    after = times[times.diff() > step].index
    blanks = pd.DataFrame(index=pd.DatetimeIndex([t - step for t in after], name=frame.index.name), columns=frame.columns)
    return pd.concat([frame, blanks]).sort_index()


class Explorer:
    """Read-only queries over the index. Loads lazily; raises IndexMissing when the index has not been built."""

    def __init__(self, root: Path = INDEX):
        self.root, self._cache = root, {}

    def _once(self, key: str, load):
        if key not in self._cache:
            self._cache[key] = load()
        return self._cache[key]

    def manifest(self) -> dict:
        def load():
            p = self.root / "manifest.json"
            if not p.exists():
                raise IndexMissing()
            return json.loads(p.read_text("utf-8"))
        return self._once("manifest", load)

    def detector(self) -> pd.DataFrame:
        self.manifest()
        return self._once("detector", lambda: pd.read_parquet(self.root / "detector.parquet"))

    def features(self) -> pd.DataFrame:
        return self._once("features", lambda: pd.read_parquet(ROOT / _cfg()[0]["preprocessing"]["features_file"]))

    def episodes(self) -> list[dict]:
        self.manifest()
        return self._once("episodes", lambda: json.loads((self.root / "episodes.json").read_text("utf-8")))

    def overview(self, bucket_hours: int = 6) -> dict:
        m, s, f = self.manifest(), self.detector(), self.features()
        metro, det = _cfg()
        start, end = f.index.min(), f.index.max()
        expected = int((end - start) / BIN) + 1
        buckets = s.resample(f"{bucket_hours}h")
        act = pd.DataFrame({"max": buckets["smoothed_score"].max(), "bins": buckets["smoothed_score"].count(),
                            "alert": buckets["alert"].apply(lambda a: bool(a.fillna(False).any()) if len(a) else False)})
        eps = self.episodes()
        return {
            "source": {"dataset": metro["dataset"]["name"], "doi": metro["dataset"]["doi"], "license": metro["dataset"]["license"],
                       "channels": {"analog": ANALOG_COLUMNS, "digital": DIGITAL_COLUMNS}, "raw_rows": m["raw_rows"], "sampling_seconds": 10,
                       "units": None},
            "span": [_iso(start), _iso(end)],
            "splits": {k: [metro["split"][k]["start"], metro["split"][k]["end"]] for k in ("train", "calibration", "test")},
            "bins": {"valid": len(f), "expected": expected, "share": round(len(f) / expected, 4), **m["bins"]},
            "threshold": m["threshold"],
            "rule": {"ewma_alpha": det["alerting"]["ewma_alpha"], "hits": det["alerting"]["persistence_hits"], "window": det["alerting"]["persistence_window"],
                     "reset_gap_minutes": det["alerting"]["reset_gap_minutes"]},
            "episodes": {"total": len(eps), "linked": sum(e["linked_incident"] is not None for e in eps),
                         "not_linked": sum(e["linked_incident"] is None for e in eps)},
            "incidents": [{"id": i["id"], "start": i["start"], "end": i["end"], "retrospective": True} for i in metro["dataset"]["reported_incidents"]],
            "activity": {"bucket_hours": bucket_hours, "t": [_iso(t) for t in act.index], "max_smoothed": _num(act["max"].tolist(), 2),
                         "bins": act["bins"].astype(int).tolist(), "alert": act["alert"].tolist()},
        }

    def range(self, start: pd.Timestamp, end: pd.Timestamp, gaps: bool = True) -> dict:
        if end <= start or end - start > MAX_RANGE:
            raise ValueError("range")
        s = self.detector().loc[start:end]
        f = self.features().loc[start:end]
        cols = [f"{c}__{k}" for c in ANALOG_COLUMNS for k in STATS] + [f"{c}__active_ratio" for c in DIGITAL_COLUMNS]
        both = s.join(f[cols], how="left")
        both = _with_gaps(both, BIN) if gaps else both  # without gap rows for day files that a static client stitches and gaps itself
        return {
            "start": _iso(start), "end": _iso(end), "threshold": self.manifest()["threshold"], "step_seconds": 300,
            "t": [_iso(t) for t in both.index],
            "split": [None if pd.isna(v) else v for v in both["split"]],
            "raw_score": _num(both["raw_score"].tolist(), 4), "smoothed_score": _num(both["smoothed_score"].tolist(), 2),
            "threshold_hit": [None if pd.isna(v) else bool(v) for v in both["threshold_hit"]],
            "alert": [None if pd.isna(v) else bool(v) for v in both["alert"]],
            "restart": [None if pd.isna(v) else bool(v) for v in both["restart"]],
            "coverage": _num(both["coverage_ratio"].tolist(), 3),
            "channels": {c: {k: _num(both[f"{c}__{k}"].tolist(), 4) for k in STATS} for c in ANALOG_COLUMNS}
                        | {c: {"active_ratio": _num(both[f"{c}__active_ratio"].tolist(), 3)} for c in DIGITAL_COLUMNS},
        }

    def raw(self, start: pd.Timestamp, end: pd.Timestamp, gaps: bool = True) -> dict:
        if end <= start or end - start > MAX_RAW:
            raise ValueError("range")
        self.manifest()
        days = pd.date_range(start.normalize(), end.normalize(), freq="D")
        frames = [pd.read_parquet(p) for d in days if (p := self.root / "raw" / f"{d.date()}.parquet").exists()]
        frame = pd.concat(frames) if frames else pd.DataFrame(columns=["timestamp", *SENSOR_COLUMNS])
        frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].set_index("timestamp")
        frame = _with_gaps(frame, pd.Timedelta(seconds=60)) if gaps else frame  # the recording is nominally every 10 s; a pause over a minute is a gap
        return {"start": _iso(start), "end": _iso(end), "step_seconds": 10, "t": [_iso(t) for t in frame.index],
                "channels": {c: _num(frame[c].tolist(), 4) for c in SENSOR_COLUMNS}}

    def bin(self, t: pd.Timestamp) -> dict:
        s = self.detector()
        if t not in s.index:
            raise KeyError("bin")
        r = s.loc[t]
        return {"t": _iso(t), "split": r["split"], "raw_score": float(r["raw_score"]), "smoothed_score": float(r["smoothed_score"]),
                "threshold": self.manifest()["threshold"], "threshold_hit": bool(r["threshold_hit"]),
                "alert": None if pd.isna(r["alert"]) else bool(r["alert"]), "restart": bool(r["restart"]),
                "coverage": float(r["coverage_ratio"]), "samples": int(r["sample_count"])}
