# AeroXAI ESIC 2026 V3 Submission Freeze Checklist

## Release branch

Create the final release branch only after all completed feature
work has been fast-forwarded into `main`.

```bash
git switch main
git pull --ff-only origin main
git status --short
git switch -c release/esic-2026-v3.1
git push -u origin release/esic-2026-v3.1
```

`git status --short` must be empty before applying release documentation.

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

- full pytest passes;
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

Open:

```text
http://127.0.0.1:5173
```

Manual browser checks:

1. **Overview** loads.
2. **Incident Replay** renders REAL detector evidence, calibration context, temporal evidence and counterfactual verification.
3. **Digital Twin Lab** runs a SIMULATED scenario.
4. **Energy Recommendation** returns an advisory action and shows no-write / re-optimization posture.
5. **Evidence Proof** renders the REAL path as withheld and the SIMULATED path as `SIMULATION_ONLY`.

### Proof-bundle API verification

With the backend running:

```bash
curl -s \
  http://127.0.0.1:8000/evidence/proof-bundle \
  -o /tmp/aeroxai-proof-bundle.json

curl -s \
  -X POST \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/aeroxai-proof-bundle.json \
  http://127.0.0.1:8000/evidence/proof-bundle/verify \
  | python -m json.tool

Evidence Proof acceptance:

```text
REAL:
physical bridge available = false
downstream physics executed = false
robust control executed = false
recommendation = null / WITHHELD

SIMULATED:
connected_to_real_incident = false
recovery passed = true
robust_safe = true
valid_for_seconds = 60
```

This browser smoke test cannot be verified from GitHub alone.

## Proof artifact integrity

These committed source artifacts must remain present:

```text
docs/evidence_action_replay.json
docs/simulated_action_proof.json
docs/proof_manifest.json
```

`docs/proof_manifest.json` must preserve:

```text
causal_claim = false
deployment.mode = advisory
override_equipment_ctrl = false
```

It must explicitly forbid:

- treating the synthetic physical state as the real MetroPT incident;
- claiming measured MetroPT energy savings from the simulated optimizer;
- claiming physical root cause from detector evidence;
- claiming identified leakage without independent process-demand information.

## Real-data evidence integrity

These reports must remain present:

```text
docs/detector_benchmark.json
docs/xai_report.json
docs/xai_verification_report.json
docs/tcn_detector_benchmark.json
docs/transformer_detector_benchmark.json
docs/raw_context_tcn_benchmark.json
```

Primary production detector remains frozen PCA.

Do not promote the raw-context TCN in submission wording beyond:

```text
exploratory secondary precursor channel
```

Required caveat:

> The same held-out test period was reused during sequential temporal-model research, so the raw-context precursor result is exploratory rather than an independent confirmatory generalization estimate.

## Simulation/control evidence integrity

These reports must remain present:

```text
docs/digital_twin_report.json
docs/baseline_controller_report.json
docs/optimizer_report.json
docs/action_explanations.json
docs/robustness_report.json
```

Do not regenerate frozen reports merely to change floating-point last digits.

## Claims integrity

Use:

```text
docs/CLAIMS.md
```

Final material must preserve:

- REAL vs MODEL_INFERENCE_FROM_REAL vs PHYSICS_MODEL_INFERENCE vs SIMULATED provenance;
- advisory-only / no PLC writes;
- PCA contribution != root cause;
- contribution percentile != fault probability;
- temporal evidence != physical fault onset;
- model-space counterfactual repair = detector-dependence evidence;
- candidate hypothesis != diagnosis;
- total outflow != leakage unless demand is independently identifiable;
- current real MetroPT physical bridge is unavailable;
- no real MetroPT compressor recommendation is issued;
- simulation-only action is not connected to the real incident;
- 0.369% = separate nominal simulated dispatch saving;
- 12.55% = separate high-leak baseline penalty;
- open-loop robustness failure motivated short-lived re-optimized advisories;
- V3 simulated first action is valid for 60 seconds under its modeled uncertainty set.

## Repository hygiene

Before final release commit:

```bash
git status --short
```

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

Use:

```text
docs/DEMO.md
```

Recommended order:

```text
Overview
-> Incident Replay
-> Evidence Proof
-> Digital Twin Lab
-> Energy Recommendation
```

## Submission narrative

The headline novelty should be:

> Proof-carrying industrial energy reasoning that refuses unsupported transitions from anomaly evidence to physical action.

Do not lead with:

- “AI compressor optimizer”;
- “Transformer anomaly detector”;
- “LLM copilot”.

Those are components or possible extensions, not the V3 identity.

## Final freeze

After all checks:

```bash
git log -1 --oneline
git status --short
```

Record the final commit SHA in submission notes.

After that, change code only for:

- reproducibility defects;
- correctness defects;
- broken runtime/demo behavior;
- submission-blocking documentation defects.
