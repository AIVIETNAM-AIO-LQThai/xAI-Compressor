# AeroXAI Submission Freeze Checklist

## Technical acceptance

Repository root:

```bash
python -m pytest -q
python -m ruff check .
```

Frontend:

```bash
cd frontend
npm ci
npm run build
```

Acceptance:
- pytest passes;
- Ruff passes;
- production frontend build passes;
- no model retuning;
- no unexpected generated files staged.

## Runtime smoke test

Backend:

```bash
python -m uvicorn backend.app.main:app --reload
```

Frontend:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`.

Only four manual interactions remain:
1. Overview loads.
2. Incident Replay switches incidents and renders attribution, calibration percentile, temporal evidence and counterfactual verification; Incident 1 explicitly shows the incomplete 10/12-bin telemetry window.
3. Digital Twin Lab runs one simulation.
4. Energy Recommendation returns one advisory action.

This browser smoke test cannot be verified from GitHub alone.

## Evidence integrity

These committed reports must remain present:

```text
docs/detector_benchmark.json
docs/xai_report.json
docs/xai_verification_report.json
docs/digital_twin_report.json
docs/baseline_controller_report.json
docs/optimizer_report.json
docs/action_explanations.json
docs/robustness_report.json
docs/evidence_summary.json
```

Do not regenerate frozen reports merely to change floating-point last digits.

## Claims integrity

Use `docs/CLAIMS.md`.

Final material must preserve:
- REAL vs SIMULATED separation;
- advisory-only / no PLC writes;
- PCA contribution != root cause;
- contribution percentile != fault probability;
- temporal evidence != physical fault onset;
- model-space counterfactual repair = detector-dependence evidence, not physical repair or causality;
- 0.369% = simulated nominal dispatch saving;
- 12.55% = separate high-leak penalty;
- open-loop robustness failure;
- 60-second recommendation validity and required re-optimization.

## Repository hygiene

Before the final submission commit:

```bash
git status --short
```

Expected: no output.

Never commit:

```text
node_modules
dist
*.tsbuildinfo
.env.local
raw MetroPT data
processed parquet datasets
```

## Demo

Use `docs/DEMO.md`.

Order:

```text
Overview
-> Incident Replay
-> Digital Twin Lab
-> Energy Recommendation
```

## Final freeze

After all checks:

```bash
git log -1 --oneline
git status --short
```

Record the final commit SHA in submission notes.

After that, change code only for a reproducibility, correctness or submission-blocking defect.
