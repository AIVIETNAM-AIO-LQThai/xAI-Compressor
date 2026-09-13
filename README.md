# AeroXAI — Explainable Compressed-Air Energy Copilot

AeroXAI is an advisory-first prototype for compressed-air energy-waste intelligence with verified explainability, developed for ESIC 2026.

The current MVP combines real historical telemetry for anomaly detection and verified detector explanation with a separate simulated physics/optimization layer for energy recommendations. The longer-term product scope is system-level compressed-air waste intelligence; the frozen MVP does not claim to diagnose or optimize every industrial waste mechanism.

The MVP never writes to a PLC or compressor controller.

## Evidence architecture

- **REAL** — MetroPT-3 telemetry for detection, incident replay, exact PCA attribution, calibration-context normality, temporal evidence and model-space counterfactual verification.
- **SIMULATED** — digital twin, baseline controller, compressor scheduling, predicted energy, action explanations and robustness stress tests.
- **LITERATURE** — external context only; never presented as AeroXAI performance.

## Frozen results

### REAL telemetry

Under the frozen chronological MetroPT-3 benchmark, the selected robust-scaled PCA detector:

- detected **4/4 documented incidents within the predefined timely window**;
- produced **pre-onset warnings for 2/4 incidents**;
- produced about **0.886 benchmark-false alert episodes per 24 h** of valid test exposure;
- achieved **PR-AUC ≈ 0.249**.

PCA contributions identify signals associated with anomaly evidence; they do **not** prove physical root cause.

### Verified XAI

The frozen detector explanation uses four explanation/verification layers followed by an explicit knowledge limit:

```text
Attribution
-> calibration-context normality
-> temporal evidence
-> model-space counterfactual verification
-> explicit knowledge limit
```

Across all four frozen incident explanations:

- the dominant contribution group was at or above approximately the **99.55th calibration-context percentile**;
- repairing only the dominant group's anomalous model-space evidence and rerunning the same EWMA / persistence pipeline caused the frozen alert to clear.

This supports a **detector-dependence** statement only. It does not establish physical root cause and does not mean that physically repairing the corresponding sensor or process variable would fix the equipment.

### SIMULATED control and energy

Frozen one-hour synthetic scenario:

- baseline energy: **25.3112 kWh**;
- optimized energy: **25.2177 kWh**;
- nominal simulated dispatch saving: **0.0935 kWh (0.369%)**;
- nominal optimized schedule: **0 modeled safety violations**.

Separate leak scenario:

- increasing leak input from 0.005 to 0.020 kg/s increased baseline energy by **3.1778 kWh (12.55%)**.

The 0.369% dispatch saving and 12.55% leak penalty are different experiments and must not be combined.

### Robustness result

The frozen nominal schedule was replayed unchanged under six non-nominal stress scenarios. Five were unsafe under the simplified model: demand +5%, demand +10%, leak +0.005 kg/s, leak +0.010 kg/s, and receiver volume -10%. The worst tested minimum pressure was **3.619 bar(g)**.

This negative result changes product behavior:

```text
deployment_mode = advisory
override_equipment_ctrl = false
recommendation_valid_for_seconds = 60
requires_reoptimization = true
open_loop_schedule_approved = false
```

## Architecture

```text
REAL TELEMETRY
MetroPT-3
  -> chronological preprocessing
  -> anomaly detection
  -> exact PCA attribution
  -> calibration-context normality
  -> temporal evidence
  -> model-space counterfactual verification
  -> knowledge limit

SIMULATED DECISION LAYER
scenario / forecast inputs
  -> physics digital twin
  -> baseline controller
  -> constrained MILP optimizer
  -> action explanation
  -> robustness / safety gate

APPLICATION
FastAPI -> React/Vite operator interface
```

## Repository

```text
backend/        FastAPI and evidence APIs
configs/        frozen detector/twin/compressor/optimizer/safety config
docs/           evidence reports, claims guardrails and demo guide
frontend/       React + TypeScript + Vite UI
ml/             preprocessing, detection, XAI, twin, control, optimization
notebooks/      executed research/validation notebooks
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
POST /twin/simulate
POST /optimization/recommend
```

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

## Evidence

Machine-readable entry point:

```text
docs/evidence_summary.json
```

Underlying reports:

```text
docs/detector_benchmark.json
docs/xai_report.json
docs/xai_verification_report.json
docs/digital_twin_report.json
docs/baseline_controller_report.json
docs/optimizer_report.json
docs/action_explanations.json
docs/robustness_report.json
```

See `docs/CLAIMS.md`, `docs/DEMO.md`, `docs/SPEC.md`, and `docs/SUBMISSION_CHECKLIST.md`.

## Status

**ESIC 2026 MVP implemented and frozen for submission preparation.**

Do not retune frozen research results unless a reproducibility or correctness defect is found.
