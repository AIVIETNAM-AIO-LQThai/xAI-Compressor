# AeroXAI Judge Demo

Recommended length: **3–3.5 minutes**.

## Start

Backend:

```bash
python -m uvicorn backend.app.main:app --reload
```

Frontend:

```bash
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## 0:00–0:25 — Overview

Say:

> AeroXAI is a proof-carrying compressed-air energy copilot. Real telemetry is used for anomaly evidence and explainability. Physics and compressor actions are only presented when their provenance supports them.

Point to:

- REAL vs SIMULATED badges;
- 4/4 timely documented-incident detection;
- 2/4 pre-onset warnings for the frozen primary PCA detector;
- advisory / no-write posture.

Do not present simulated savings as measured factory savings.

## 0:25–1:10 — Verified Incident Replay

Use **Incident 2**.

Key evidence:

```text
warning timing:             2.17 h before documented onset
dominant group:             dv_pressure
contribution share:         about 77.6%
calibration percentile:     about 99.55th
temporal window:            12 / 12 bins
smoothed score:             about 58,989
after model-space repair:   about 7,472
frozen threshold:           about 52,942
result:                     alert clears
```

Say:

> AeroXAI first tells us what contributed to the detector decision. DV pressure accounts for about 77.6% of this alert's evidence.

Then:

> We compare that contribution with calibration behavior. It is around the 99.55th percentile, meaning this detector evidence is highly unusual relative to calibration.

Point to the temporal heatmap:

> This shows how detector evidence developed over past observations. It is not a physical fault-onset estimator.

Point to Counterfactual Verification:

> We repair only the dominant anomalous evidence in PCA feature space and rerun the unchanged alert pipeline. The frozen alert clears.

Then explicitly say:

> That verifies detector dependence. It does not prove DV pressure physically caused the incident.

## 1:10–1:55 — Evidence Proof

Open **Evidence Proof**.

Start with the REAL panel.

Say:

> This is the V3 difference. AeroXAI does not jump from an anomaly to a compressor command. Real detector evidence generates multiple candidate hypotheses, then the physical-bridge gate asks whether the asset is calibrated well enough to support real physical reasoning.

Point to:

```text
Physical bridge withheld
Physics executed: NO
Robust control executed: NO
Real recommendation: WITHHELD
```

Say:

> For MetroPT, the required site-calibrated receiver/compressor model and physical observations are not available, so AeroXAI stops here rather than inventing a leak rate or action.

Move to the SIMULATED panel.

Say:

> Separately, we validate the downstream reasoning chain on known synthetic physics. The inverse model recovers the known total outflow, propagates a bounded uncertainty set, and one shared compressor action is checked across all three scenarios.

Point to:

```text
recovery PASS
scenario count 3
validity 60 s
fixed_1 ON
fixed_2 OFF
vsd_1 20%
ROBUST SAFE
```

Then point to:

> Not connected to the MetroPT incident.

Say:

> The two proof paths are shown together for product clarity, but they remain scientifically separate.

## 1:55–2:20 — Digital Twin Lab

Run the balanced default:

```text
initial pressure 7.0 bar(g)
inflow 0.095 kg/s
demand 0.090 kg/s
leak 0.005 kg/s
```

Pressure should remain approximately constant.

Change leak to `0.020 kg/s` and run again.

Say:

> This page is simulated physics. The leak value is a scenario input, not an inferred MetroPT leak flow.

## 2:20–2:55 — Energy Recommendation

Generate the default advisory action.

Say:

> The product remains advisory-only. The earlier deterministic optimizer demonstrates the nominal energy objective, while the V3 proof layer adds uncertainty-aware shared-action optimization and a first-action safety gate. No open-loop schedule is approved for unattended execution.

Point to:

- recommendation validity;
- reserve / pressure rationale;
- no equipment write;
- re-optimization requirement.

## 2:55–3:15 — Close

> AeroXAI is not just an anomaly detector and not just an optimizer. It is a proof-carrying reasoning system: detect on real telemetry, verify the detector evidence, preserve competing hypotheses, refuse unsupported physical claims, validate actions through physics and robust optimization, and carry the evidence boundary all the way to the operator interface.

## Research question if asked

If a judge asks about deep learning:

> We benchmarked causal TCN and Transformer representations rather than assuming a deeper model was better. The raw 10-second TCN showed 4/4 pre-onset alerts on its valid retrospective subset, but it also produced many more false alerts than aligned PCA. We therefore kept PCA as the primary detector and treat the raw TCN only as an exploratory precursor channel pending independent validation.

## Safe numbers to quote

REAL primary PCA:

- 4/4 documented incidents detected within the timely window;
- 2/4 pre-onset warnings;
- false alert episodes about 0.886 per 24 h;
- PR-AUC about 0.249.

Verified XAI:

- dominant explanation group at or above about the 99.55th calibration-context percentile in all four frozen explanations;
- dominant-group model-space repair cleared all four frozen alerts under the unchanged alert pipeline.

Exploratory raw-context TCN:

- 4/4 pre-onset on its valid subset;
- false alerts about 2.579 per 24 h;
- PR-AUC about 0.233;
- not production-promoted.

Simulation-only V3 proof:

- known total outflow 0.110 kg/s;
- bounded scenarios about 0.106 / 0.110 / 0.114 kg/s;
- leakage not identifiable without independent demand;
- robust-safe 60-second first action across three modeled scenarios;
- fixed_1 ON, fixed_2 OFF, VSD 20%.

Earlier nominal simulated benchmark:

- baseline 25.3112 kWh;
- optimized 25.2177 kWh;
- 0.369% nominal dispatch energy reduction;
- separate high-leak penalty 12.55%.

Never combine the 12.55% high-leak penalty with the 0.369% optimizer saving.

## If asked why the real recommendation is withheld

> That is the point of the proof-carrying architecture. A recommendation should not become more certain simply because the software can calculate one. Until the asset has the calibrated physical model and observations required by the inverse model, AeroXAI keeps the real evidence path at hypothesis level and does not issue a real compressor action.
