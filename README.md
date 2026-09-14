# AeroXAI — Proof-Carrying Explainable Compressed-Air Energy Copilot

AeroXAI is an advisory-first prototype for compressed-air energy-waste intelligence, developed for ESIC 2026.

The V3 prototype is built around a simple rule:

> **Data proposes evidence; physics and optimization verify what can safely be claimed.**

AeroXAI uses real historical telemetry for anomaly detection and verified detector explanation, then generates candidate waste hypotheses. Before any real-asset physical state or control recommendation can be issued, a physical-bridge gate checks whether the asset has the required calibrated model and observations. In the current MetroPT prototype that bridge is intentionally **withheld**, so no real-asset physics or action is fabricated.

Separately, a simulation-only validation path tests inverse physics, bounded uncertainty, robust optimization, compressor runtime continuity, and a short-lived safety-gated advisory.

The MVP never writes to a PLC or compressor controller.

## Product identity

**AeroXAI is a proof-carrying industrial energy reasoning system that turns anomalous telemetry into physically tested hypotheses and robust, auditable actions — while making the evidence boundary visible.**

The current prototype demonstrates two intentionally separate paths:

```text
REAL EVIDENCE PATH
MetroPT-3 telemetry
  -> chronological anomaly detection
  -> verified XAI
  -> candidate waste hypotheses
  -> physical-bridge gate
  -> WITHHOLD real-asset action
     because site calibration / physical observations are absent

SIMULATION-ONLY VALIDATION PATH
known synthetic physical truth
  -> synthetic pressure trajectory
  -> inverse physics
  -> bounded physical-state uncertainty
  -> scenario generation
  -> shared-action robust MILP
  -> independent first-action safety gate
  -> 60-second advisory
  -> proof manifest
```

The two paths coexist in one product but are never blended into one scientific claim.

## Provenance model

AeroXAI uses the following provenance labels:

- **REAL** — measured MetroPT-3 telemetry and frozen real-data evaluation.
- **MODEL_INFERENCE_FROM_REAL** — candidate hypotheses generated from verified real detector evidence; not diagnoses.
- **PHYSICS_MODEL_INFERENCE** — physical quantities inferred from the receiver model; not direct measurements.
- **SIMULATED** — synthetic digital-twin, optimization, safety-gate and energy results.
- **LITERATURE** — external context only.

All V3 reasoning outputs keep `causal_claim = false`.

## Frozen primary real-data result

The production detection baseline remains **RobustScaler + PCA reconstruction error** with causal EWMA smoothing and persistence.

Under the frozen chronological MetroPT-3 benchmark:

- **4/4** documented incidents were detected within the predefined timely window;
- **2/4** produced pre-onset warnings;
- benchmark-false alert episodes were about **0.886 per 24 h** of valid exposure;
- **PR-AUC ≈ 0.249**.

PCA contributions identify signals contributing to anomaly evidence; they do **not** establish physical root cause.

## Verified XAI

The frozen explanation pipeline is:

```text
exact PCA contribution
-> calibration-context rarity
-> past-only temporal evidence
-> model-space counterfactual detector-dependence test
-> explicit knowledge limit
```

Across all four frozen incident explanations:

- the dominant contribution group was at or above approximately the **99.55th calibration-context percentile**;
- repairing only the dominant anomalous model-space evidence and rerunning the unchanged EWMA / persistence pipeline cleared the frozen alert.

This is detector-dependence evidence only. It is not a physical intervention and does not prove root cause.

## Temporal representation research

AeroXAI also benchmarks causal deep temporal representations as research channels. None replaced the frozen PCA detector.

| Model | Timely recall | Pre-onset recall | False alerts / 24 h | PR-AUC | Status |
|---|---:|---:|---:|---:|---|
| Frozen PCA reference | 1.00 | 0.50 | 0.886 | 0.249 | Primary detector |
| 5-min causal TCN | 1.00 | 0.75 | 2.009 | 0.096 | Not promoted |
| Causal Transformer | 1.00 | 0.50 | 1.753 | 0.165 | Not promoted |
| Raw-context causal TCN | 1.00 | **1.00** | 2.579 | 0.233 | Exploratory precursor channel |

For model-vs-PCA comparisons, use the aligned PCA metrics in `docs/RESEARCH_RESULTS.md`; valid target timestamps differ across experiments.

The raw-context TCN produced pre-onset alerts for all four documented incidents on its valid test subset, but also produced substantially more false alerts than aligned PCA. Because the same held-out test period was revisited across temporal-model experiments, this precursor result is **exploratory rather than independent confirmatory evidence**.

## Real evidence-to-action gate

For the current MetroPT prototype, the real path ends with:

```text
WITHHOLD_REAL_ASSET_ADVISORY_MISSING_VALIDATED_PHYSICAL_BRIDGE
```

Missing requirements:

- site-calibrated receiver and compressor model;
- calibrated compressor inflow observation;
- receiver pressure trace appropriate for the inverse model.

Therefore:

```text
downstream_physics_executed = false
robust_control_executed = false
recommendation = null
```

This refusal is a product behavior, not a missing demo feature.

## Simulation-only physics and robust action proof

A separate synthetic validation uses known total outflow of **0.110 kg/s**.

Results:

- inverse-physics recovery passed;
- inferred mean total outflow: **0.11000000000000001 kg/s**;
- absolute numerical recovery error: about **1.39e-17 kg/s**;
- bounded total-outflow scenarios: approximately **0.106 / 0.110 / 0.114 kg/s**;
- leakage remains **unidentifiable** because independent process demand is absent;
- shared-action robust MILP evaluates all three scenarios together;
- selected 60-second action passes the independent first-action safety gate.

Current simulated first action:

```text
fixed_1  ON
fixed_2  OFF
vsd_1    20%
```

For that bounded synthetic scenario set:

- worst first-interval pressure minimum ≈ **6.805 bar(g)**;
- worst first-interval pressure maximum ≈ **6.846 bar(g)**;
- minimum modeled reserve ≈ **0.026 kg/s**;
- recommendation validity = **60 s**.

This is **SIMULATION_ONLY** and is explicitly not connected to a real MetroPT incident.

## Earlier frozen energy benchmarks

A separate one-hour nominal synthetic scheduling benchmark remains part of the evidence pack:

- baseline energy: **25.3112 kWh**;
- optimized energy: **25.2177 kWh**;
- nominal simulated dispatch saving: **0.0935 kWh (0.369%)**;
- nominal optimized schedule: **0 modeled safety violations**.

A separate high-leak baseline scenario increased predicted one-hour energy by **3.1778 kWh (12.55%)** when leak input changed from 0.005 to 0.020 kg/s.

The **0.369% dispatch saving** and **12.55% high-leak penalty** are different experiments and must never be combined.

The earlier frozen one-hour open-loop schedule also failed several non-nominal replay tests, which motivated the V3 short-lived robust receding-horizon action gate.

## Architecture

```text
REAL TELEMETRY
MetroPT-3
  -> frozen PCA detector
  -> verified XAI
  -> candidate hypothesis engine
  -> physical-bridge gate
  -> withhold if required calibration/observations are absent

TEMPORAL RESEARCH CHANNEL
5-min causal TCN / causal Transformer / raw-context TCN
  -> benchmark only
  -> no production promotion unless evidence supports it

SIMULATED PHYSICS + CONTROL
synthetic physical truth
  -> receiver inverse physics
  -> explicit identifiability test
  -> bounded uncertainty
  -> physical scenario set
  -> state-aware shared-action robust MILP
  -> independent first-action safety gate
  -> 60-second advisory
  -> runtime-state update
  -> re-observe / re-optimize

PROOF LAYER
real evidence replay
+ simulated action proof
-> machine-readable proof manifest
-> SHA-256-bound proof bundle
-> source-artifact integrity verification
-> semantic manifest rebuild
-> API
-> Evidence Proof UI
```

## Repository

```text
backend/        FastAPI, evidence APIs and proof manifest
configs/        detector/twin/compressor/optimizer/bridge configs
docs/           evidence reports, proof files, claims and demo guide
frontend/       React + TypeScript + Vite operator interface
ml/             detection, XAI, temporal models, reasoning, twin, optimization
notebooks/      research/validation notebooks
scripts/        reproducible report/proof generation
tests/          unit, acceptance and API tests
```

## Run

Python 3.12 recommended:

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload
```

Backend: `http://127.0.0.1:8000`  
Swagger: `http://127.0.0.1:8000/docs`

Node 20 recommended:

```bash
cd frontend
npm ci
npm run dev
```

Frontend: `http://127.0.0.1:5173`

## Main API

```text
GET  /health
GET  /evidence/detection
GET  /evidence/summary
GET  /evidence/proof-manifest
GET  /evidence/proof-bundle
POST /evidence/proof-bundle/verify
POST /twin/simulate
POST /optimization/recommend
```

The proof bundle embeds the frozen REAL evidence replay and
SIMULATED action proof together with SHA-256 fingerprints.
The verifier checks the bundle ID, manifest digest, embedded
artifact digests, current source-file hashes, and whether the
embedded source artifacts rebuild the manifest exactly.

This integrity mechanism verifies provenance and reproducibility.
It does not turn simulated evidence into real evidence and does
not establish physical causality.

There is intentionally no PLC/control-write endpoint.

## Validation

Repository root:

```bash
python -m pytest -q
python -m ruff check .
```

Frontend:

```bash
cd frontend
npm run build
```

## Key machine-readable evidence

```text
docs/detector_benchmark.json
docs/xai_verification_report.json
docs/tcn_detector_benchmark.json
docs/transformer_detector_benchmark.json
docs/raw_context_tcn_benchmark.json
docs/evidence_action_replay.json
docs/simulated_action_proof.json
docs/proof_manifest.json
```

See `docs/CLAIMS.md`, `docs/RESEARCH_RESULTS.md`, `docs/DEMO.md`, `docs/SPEC.md`, and `docs/SUBMISSION_CHECKLIST.md`.

## Status

**ESIC 2026 V3 release candidate.**

Do not retune frozen research results unless a reproducibility or correctness defect is found.
