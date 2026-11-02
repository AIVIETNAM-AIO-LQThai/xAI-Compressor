"""Computed evidence for one alert episode, using the frozen detector's own explanation code.

For an episode with no reviewed case file this gives what the detector itself can say at the first alert bin: how the 3-of-4 rule was met,
which signal groups carry the smoothed score and where each sits against the calibration period, and how those percentiles moved over the
preceding hour. It is computed on request, not reviewed, and makes no causal claim. It does not run counterfactual repairs, rank candidate
explanations or use any document. The same code reproduces the frozen incident figures in docs/xai_verification_report.json (tested).
"""
import numpy as np
import pandas as pd

from backend.app.explorer import BIN, ROOT, Explorer, _cfg, _iso
from ml.data.features import model_feature_columns
from ml.detection.pca_detector import fit_pca_detector
from ml.explainability.pca import (
    aggregate_contributions,
    causal_ewma_contributions,
    explain_pca_detector,
)
from ml.explainability.verification import (
    contribution_percentiles,
    fit_contribution_reference,
    temporal_evidence_window,
)

WINDOW_BINS = 12  # the frozen verification's evidence window: one hour of five-minute bins


class EpisodeEvidence:
    def __init__(self, explorer: Explorer):
        self.ex = explorer
        self._model = None

    def _load(self):
        if self._model is None:
            metro, det = _cfg()
            pre, al = metro["preprocessing"], det["alerting"]
            frames = {n: pd.read_parquet(ROOT / pre[f"{n}_file"]) for n in ("train", "calibration", "test")}
            detector = fit_pca_detector(frames["train"], model_feature_columns(frames["train"].columns),
                                        variance_retained=det["pca"]["variance_retained"], scaler_name=det["pca"]["scaler"])
            smooth = {n: causal_ewma_contributions(aggregate_contributions(explain_pca_detector(frames[n], detector).contributions),
                                                   alpha=al["ewma_alpha"], reset_gap_minutes=al["reset_gap_minutes"]) for n in ("calibration", "test")}
            reference = fit_contribution_reference(smooth["calibration"])
            self._model = (smooth["test"], contribution_percentiles(smooth["test"], reference), al)
        return self._model

    def at(self, t: pd.Timestamp) -> dict:
        """The evidence at one alert bin of the test period."""
        smoothed, percentiles, al = self._load()
        series = self.ex.detector()
        if t not in smoothed.index or t not in series.index:
            raise KeyError("bin")
        row = smoothed.loc[t].astype(float)
        total = float(row.sum())
        groups = [{"group": str(g), "smoothed_contribution": float(row[g]), "share": float(row[g]) / total, "calibration_percentile": float(percentiles.loc[t, g])}
                  for g in sorted(row.index, key=lambda g: row[g], reverse=True)]

        # The rule's own window: the last four bins up to this one, starting again after a smoothing restart.
        recent = series.loc[:t].tail(al["persistence_window"])
        restarts = np.flatnonzero(recent["restart"].to_numpy())
        recent = recent.iloc[restarts[-1]:] if len(restarts) else recent
        now = series.loc[t]
        alpha = al["ewma_alpha"]
        carried = None if now["restart"] else (1 - alpha) * float(series["smoothed_score"].shift(1).loc[t])
        window = temporal_evidence_window(percentiles, end_time=t, window_bins=WINDOW_BINS, bin_minutes=int(BIN.total_seconds() // 60))
        return {
            "time": _iso(t),
            "threshold": self.ex.manifest()["threshold"],
            "rule": {"hits": al["persistence_hits"], "window": al["persistence_window"], "alpha": alpha},
            "smoothed_score": float(now["smoothed_score"]),
            "raw_score": float(now["raw_score"]),
            "restart": bool(now["restart"]),
            "carried_share": None if carried is None else carried / float(now["smoothed_score"]),
            "recent": [{"time": _iso(i), "smoothed_score": float(r["smoothed_score"]), "hit": bool(r["threshold_hit"]), "restart": bool(r["restart"])} for i, r in recent.iterrows()],
            "groups": groups,
            "window": {"complete": len(window) == WINDOW_BINS, "bins": [{"time": _iso(i), "percentiles": {g: float(v) for g, v in r.items()}} for i, r in window.iterrows()]},
        }

    def episode(self, start: pd.Timestamp) -> dict:
        found = next((e for e in self.ex.episodes() if pd.Timestamp(e["start"]) == start), None)
        if found is None:
            raise KeyError("episode")
        incident = found["linked_incident"]
        return {
            "episode": found,
            "status": "computed_not_reviewed",
            "causal_claim": False,
            "not_computed": ["counterfactual_repairs", "candidate_explanations", "documents"],
            "linked_incident": incident,
            "evidence": self.at(start),
        }
