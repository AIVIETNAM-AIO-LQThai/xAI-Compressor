"""Scenario lab: declared, seeded changes that are always labelled as simulated.

Two models, neither of which is a MetroPT-3 physics model:

- sensor faults on a recorded background. A recorded 10-second window from the test period gets one declared change to one channel, then the
  unchanged pipeline runs (five-minute features, the frozen scaler and PCA, EWMA, the frozen threshold, the 3-of-4 rule). The same window
  without the change is scored alongside it. The detector is never told what was injected and the lab does not claim it identifies it;
- a generic receiver twin: the existing isothermal receiver and baseline controller with a seeded demand and a declared demand step or
  leak. The twin cannot produce MetroPT-3 channels, so the detector is not applicable to it.

Leaks, demand changes and compressor degradation are not offered on the recorded background: that would need a validated MetroPT-3 physics
model that does not exist here.
"""
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from backend.app.explorer import BIN, ROOT, Explorer, _cfg, _iso
from backend.app.runtime import load_runtime_config
from ml.control.baseline import BaselineControllerConfig, simulate_baseline
from ml.data.features import (
    aggregate_causal_bins,
    filter_valid_bins,
    model_feature_columns,
)
from ml.data.schema import ANALOG_COLUMNS, DIGITAL_COLUMNS
from ml.detection.alerts import causal_ewma, persistent_alerts
from ml.detection.pca_detector import fit_pca_detector, score_pca_detector

WARMUP = pd.Timedelta(hours=4)  # recorded, unchanged history before the window, so smoothing and the alert rule start where the recording was
ANALOG_FAULTS = ("offset", "drift", "flatline", "noise")


class SensorFault(BaseModel):
    start: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}$")
    hours: int = Field(ge=6, le=24)
    fault: Literal["offset", "drift", "flatline", "noise", "dropout", "stuck"]
    channel: str | None = None
    onset_hours: float = Field(ge=1, le=23)
    duration_hours: float | None = Field(default=None, gt=0, le=24)
    severity: float = Field(default=1.0, ge=0, le=20)  # offset, drift, noise: multiples of the channel's training standard deviation; dropout: share of samples lost
    stuck_value: Literal[0, 1] = 0
    seed: int = Field(default=0, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def check(self):
        if self.onset_hours >= self.hours:
            raise ValueError("onset_hours must fall inside the window")
        if self.fault in ANALOG_FAULTS and self.channel not in ANALOG_COLUMNS:
            raise ValueError("this fault needs an analog channel")
        if self.fault == "stuck" and self.channel not in DIGITAL_COLUMNS:
            raise ValueError("a stuck-at fault needs a digital channel")
        if self.fault == "dropout":
            if self.severity > 1:
                raise ValueError("dropout severity is a share between 0 and 1")
            self.channel = None
        return self


class TwinRun(BaseModel):
    hours: float = Field(default=1.0, ge=0.25, le=6)
    base_demand_kg_s: float = Field(default=0.09, ge=0.02, le=0.15)
    noise_percent: float = Field(default=2.0, ge=0, le=10)
    scenario: Literal["none", "demand_step", "leak_onset"] = "none"
    onset_minutes: float = Field(default=20, ge=1, le=350)
    duration_minutes: float | None = Field(default=None, gt=0, le=360)  # absent: to the end of the run
    magnitude_kg_s: float = Field(default=0.02, ge=0, le=0.05)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def check(self):
        if self.onset_minutes >= self.hours * 60:
            raise ValueError("onset_minutes must fall inside the run")
        return self


class Lab:
    def __init__(self, explorer: Explorer):
        self.ex = explorer
        self._m = None

    def _model(self):
        if self._m is None:
            metro, det = _cfg()
            pre = metro["preprocessing"]
            train = pd.read_parquet(ROOT / pre["train_file"])
            feats = model_feature_columns(train.columns)
            detector = fit_pca_detector(train, feats, variance_retained=det["pca"]["variance_retained"], scaler_name=det["pca"]["scaler"])
            days = pd.date_range(metro["split"]["train"]["start"], metro["split"]["train"]["end"], freq="D").normalize()
            raw = pd.concat([pd.read_parquet(p) for d in days if (p := self.ex.root / "raw" / f"{d.date()}.parquet").exists()])
            std = {c: float(raw[c].std()) for c in ANALOG_COLUMNS}
            self._m = (detector, {f: (float(train[f].min()), float(train[f].max())) for f in feats}, std, det["alerting"], pre, metro["split"]["test"])
        return self._m

    def _raw(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        days = pd.date_range(start.normalize(), end.normalize(), freq="D")
        frames = [pd.read_parquet(p) for d in days if (p := self.ex.root / "raw" / f"{d.date()}.parquet").exists()]
        if not frames:
            raise KeyError("raw")
        f = pd.concat(frames)
        f = f[(f.timestamp > start) & (f.timestamp <= end)].reset_index(drop=True)
        return f.astype({c: "float64" for c in f.columns if c != "timestamp"})  # the index stores float32; features are computed in double

    def _score(self, raw: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp):
        detector, ranges, _, al, pre, _ = self._model()
        feats = filter_valid_bins(aggregate_causal_bins(raw, bin_minutes=pre["bin_minutes"], expected_samples=pre["expected_samples_per_bin"]), minimum_coverage=pre["minimum_coverage"])
        s = score_pca_detector(feats, detector)
        smooth = causal_ewma(s, alpha=al["ewma_alpha"], reset_gap_minutes=al["reset_gap_minutes"])
        thr = self.ex.manifest()["threshold"]
        _, alert = persistent_alerts(smooth, threshold=thr, required_hits=al["persistence_hits"], window_bins=al["persistence_window"], reset_gap_minutes=al["reset_gap_minutes"])
        inside = (feats.index > lo) & (feats.index <= hi)
        out = {f: int(((feats.loc[inside, f] < a) | (feats.loc[inside, f] > b)).sum()) for f, (a, b) in ranges.items()}
        grid = pd.date_range(lo + BIN, hi, freq=BIN)
        frame = pd.DataFrame({"raw": s, "smoothed": smooth, "hit": smooth >= thr, "alert": alert}).reindex(grid)
        return frame, feats.reindex(grid), {f: n for f, n in out.items() if n}

    def _inject(self, raw: pd.DataFrame, r: SensorFault, onset: pd.Timestamp) -> pd.DataFrame:
        out = raw.copy()
        end = onset + pd.Timedelta(hours=r.duration_hours) if r.duration_hours else out.timestamp.max() + pd.Timedelta(seconds=1)
        on = (out.timestamp >= onset) & (out.timestamp < end)
        rng = np.random.default_rng(r.seed)
        if r.fault == "dropout":
            return out[~(on & (rng.random(len(out)) < r.severity))].reset_index(drop=True)
        c = r.channel
        if r.fault == "stuck":
            out.loc[on, c] = r.stuck_value
        elif r.fault == "flatline":
            before = out.loc[out.timestamp < onset, c]
            out.loc[on, c] = before.iloc[-1] if len(before) else out.loc[on, c].iloc[0]
        else:
            step = r.severity * self._model()[2][c]
            if r.fault == "offset":
                out.loc[on, c] += step
            elif r.fault == "drift":
                span = (end - onset).total_seconds()
                out.loc[on, c] += step * ((out.timestamp[on] - onset).dt.total_seconds() / span).clip(0, 1)
            else:
                out.loc[on, c] += rng.normal(0, step, int(on.sum()))
        return out

    def backgrounds(self, hours: int = 12, count: int = 8) -> list[dict]:
        """Recorded windows with no threshold crossing and nearly complete coverage, evenly spread over the test period: plain backgrounds to perturb."""
        s = self.ex.detector()
        t = s[s.split == "test"]
        step = pd.Timedelta(hours=hours)
        out, at = [], t.index.min().ceil("h") + WARMUP
        while at + step <= t.index.max():
            w = t.loc[at:at + step]
            if len(w) >= 0.97 * step / BIN and not w.threshold_hit.any() and len(t.loc[at - WARMUP:at]) >= 0.95 * WARMUP / BIN and not t.loc[at - WARMUP:at].alert.any():
                out.append(at)
                at += pd.Timedelta(days=21)
            else:
                at += pd.Timedelta(hours=6)
        return [{"start": _iso(a)[:16].replace(" ", "T"), "hours": hours} for a in out[:count]]

    def sensor_fault(self, r: SensorFault) -> dict:
        _, ranges, std, _, _, test = self._model()
        lo = pd.Timestamp(r.start.replace("T", " "))
        hi = lo + pd.Timedelta(hours=r.hours)
        if lo - WARMUP < pd.Timestamp(test["start"]) or hi > pd.Timestamp(test["end"]):
            raise ValueError("window")
        raw = self._raw(lo - WARMUP, hi)
        if raw.empty or len(raw) < 0.5 * (r.hours + 4) * 360:
            raise ValueError("coverage")
        onset = lo + pd.Timedelta(hours=r.onset_hours)
        base, base_f, base_out = self._score(raw, lo, hi)
        hurt, hurt_f, hurt_out = self._score(self._inject(raw, r, onset), lo, hi)
        t = [_iso(i) for i in base.index]
        col = f"{r.channel}__{'mean' if r.channel in ANALOG_COLUMNS else 'active_ratio'}" if r.channel else None
        num = lambda v: [None if pd.isna(x) else round(float(x), 4) for x in v]

        def series(f):
            return {"raw_score": num(f["raw"]), "smoothed_score": num(f["smoothed"]), "threshold_hit": [None if pd.isna(x) else bool(x) for x in f["hit"]],
                    "alert": [None if pd.isna(x) else bool(x) for x in f["alert"]]}

        def first(f):
            hit = f.index[f["alert"].fillna(False).astype(bool)]
            return _iso(hit[0]) if len(hit) else None

        fault_end = onset + pd.Timedelta(hours=r.duration_hours) if r.duration_hours else hi
        return {
            "evidence_class": "SIMULATED",
            "model": "recorded_background_sensor_fault",
            "window": {"start": _iso(lo), "end": _iso(hi), "warmup_hours": 4},
            "injected": {"fault": r.fault, "channel": r.channel, "onset": _iso(onset), "end": _iso(fault_end), "severity": r.severity,
                         "severity_unit": {"dropout": "share of samples lost", "flatline": None, "stuck": None}.get(r.fault, "training standard deviations"),
                         "stuck_value": r.stuck_value if r.fault == "stuck" else None, "seed": r.seed,
                         "training_std": std.get(r.channel) if r.channel in std else None},
            "threshold": self.ex.manifest()["threshold"],
            "t": t,
            "baseline": series(base),
            "perturbed": series(hurt),
            "channel": None if col is None else {"name": r.channel, "feature": col, "baseline": num(base_f[col]), "perturbed": num(hurt_f[col])},
            "first_alert": {"baseline": first(base), "perturbed": first(hurt)},
            "bins": {"expected": len(t), "baseline_valid": int(base["raw"].notna().sum()), "perturbed_valid": int(hurt["raw"].notna().sum())},
            "support": {"baseline": _top(base_out), "perturbed": _top(hurt_out), "training_features": len(ranges)},
            "detector_identifies_fault": False,
            "notes": ["The change is declared and applied to recorded samples; it is not a physical model of how a real fault would appear on MetroPT-3.",
                      "The detector sees only the resulting samples. It is not told what was injected and does not name a fault."],
        }

    def twin(self, r: TwinRun) -> dict:
        """Separate synthetic inputs (base demand, added demand, leakage) and the plant's response to their sum.

        The controller and the receiver in this model depend only on total outflow (the controller's set point uses demand + leak, the
        receiver uses inflow - demand - leak), so the unchanged controller is driven with total outflow as its demand profile. The
        decomposition is kept as ground truth in the response; nothing observable in this model carries it."""
        rt = load_runtime_config()
        p = rt.pressure
        cfg = BaselineControllerConfig(target_bar_g=p["target_bar_g"], lower_band_bar_g=p["lower_band_bar_g"], upper_band_bar_g=p["upper_band_bar_g"],
                                       safety_min_bar_g=p["safety_min_bar_g"], safety_max_bar_g=p["safety_max_bar_g"], pressure_gain_kg_s_per_bar=0.025)
        n = int(r.hours * 3600)
        rng = np.random.default_rng(r.seed)
        base = np.clip(r.base_demand_kg_s * (1 + r.noise_percent / 100 * rng.standard_normal(n)), 0, None)
        onset = int(r.onset_minutes * 60)
        end = min(n, onset + int(r.duration_minutes * 60)) if r.duration_minutes else n
        step = np.zeros(n)
        if r.scenario != "none":
            step[onset:end] = r.magnitude_kg_s
        added, leak = (step, np.zeros(n)) if r.scenario == "demand_step" else (np.zeros(n), step)
        total = base + added + leak
        d = simulate_baseline(list(total), leak_mass_flow_kg_s=0.0, initial_pressure_bar_g=p["target_bar_g"], parameters=rt.twin_parameters,
                              compressors=rt.compressors, controller_config=cfg, timestep_seconds=1.0)
        at = np.arange(9, n, 10)  # one point every ten seconds; flows are the values at that second, so total = parts holds point by point
        k = d.iloc[at]
        unsafe = d.safety_violation.to_numpy().astype(bool)
        changed = r.scenario != "none"
        ids = [c.id for c in rt.compressors]
        r5 = lambda v: np.round(v, 6).tolist()
        return {
            "evidence_class": "SIMULATED",
            "source": "SIMULATED · generic plant",
            "model": "generic_receiver_twin",
            "detector": "not_applicable",
            "detector_reason": "The twin has no MetroPT-3 channels, so the MetroPT-3 detector cannot score it.",
            "scenario": r.scenario, "seed": r.seed, "magnitude_kg_s": r.magnitude_kg_s if changed else 0.0,
            "onset_seconds": onset if changed else None, "end_seconds": end if changed else None,
            "units": {"time": "s from the start of the run", "flow": "kg/s", "pressure": "bar g", "power": "kW", "energy": "kWh"},
            "band": {"lower": p["lower_band_bar_g"], "upper": p["upper_band_bar_g"], "safety_min": p["safety_min_bar_g"], "safety_max": p["safety_max_bar_g"]},
            "time_seconds": (at + 1).tolist(),
            "inputs": {"base_demand_kg_s": r5(base[at]), "added_demand_kg_s": r5(added[at]), "leak_kg_s": r5(leak[at]), "total_outflow_kg_s": r5(total[at])},
            "response": {"pressure_bar_g": k.pressure_bar_g.round(4).tolist(), "compressor_flow_kg_s": k.compressor_flow_kg_s.round(6).tolist(),
                         "power_kw": k.power_kw.round(3).tolist(), "safety_violation": [bool(unsafe[i - 9:i + 1].any()) for i in at],  # any violation in the ten seconds the point closes
                         "on": {i: [bool(x) for x in k[f"{i}_on"]] for i in ids}},
            "summary": {"pressure_min": float(d.pressure_bar_g.min()), "pressure_max": float(d.pressure_bar_g.max()), "energy_kwh": float(d.energy_kwh.sum()),
                        "safety_violations": int(unsafe.sum()), "starts": int(sum(d[c].sum() for c in d.columns if c.endswith("_start"))),
                        "base_demand_kg": float(base.sum()), "added_demand_kg": float(added.sum()), "leak_kg": float(leak.sum()), "total_outflow_kg": float(total.sum())},
            "notes": ["Added demand is useful consumption; leakage is outflow that does no work. In this model the controller and the receiver respond only to their sum, so equal totals give identical pressure, compressor flow, power and energy. That is a property of this simplified plant observed through pressure and total flow, not a claim about every real plant or every measurement.",
                      "A generic 10 m3 isothermal receiver with the fleet in configs/compressors.yaml; not a model of the MetroPT-3 compressor. Energy is simulated, not a measured saving."],
        }


def _top(out: dict[str, int], n: int = 8) -> dict:
    return {"features_outside": len(out), "top": sorted(({"feature": f, "bins": c} for f, c in out.items()), key=lambda x: -x["bins"])[:n]}
