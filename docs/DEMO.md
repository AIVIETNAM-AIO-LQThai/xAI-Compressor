# AeroXAI Judge Demo

Recommended length: **2.5–3 minutes**.

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

Open `http://127.0.0.1:5173`.

## 0:00–0:35 — Overview

Say:

> AeroXAI is an explainable compressed-air energy copilot. Real telemetry is used for detection and explainability; energy optimization remains explicitly simulated.

Point to REAL vs SIMULATED badges, 4/4 timely detections, 2/4 pre-onset warnings, 0.369% nominal simulated dispatch saving, and the advisory/re-optimization posture.

Do not present 0.369% as measured factory savings.

## 0:35–1:15 — Incident Replay

Choose Incident 2 or 4.

Incident 2: about 2.17 h pre-onset.  
Incident 4: about 5.75 h pre-onset.

Say:

> The contribution chart shows which signal groups contributed to the same smoothed PCA alert score used by the detector. It is not a physical root-cause diagnosis.

## 1:15–1:55 — Digital Twin Lab

Run the balanced default:

```text
initial pressure 7.0 bar(g)
inflow 0.095 kg/s
demand 0.090 kg/s
leak 0.005 kg/s
duration 600 s
```

Pressure should remain approximately constant.

Then change only leak to `0.020 kg/s` and run again.

Say:

> This is simulated physics evidence. The leak value is a scenario input, not an inferred MetroPT leak flow.

## 1:55–2:40 — Energy Recommendation

Generate the default advisory action.

Expected initial nominal action:

```text
fixed_1 ON
fixed_2 OFF
vsd_1 20%
```

Point to predicted pressure, safety margin, reserve, explanation, `valid 60s`, and `Open-loop horizon is not approved`.

Say:

> The optimizer solves the forecast horizon, but AeroXAI surfaces only the current 60-second advisory action. Robustness testing showed that the frozen one-hour open-loop schedule can become unsafe under mismatch, so re-optimization is mandatory.

## 2:40–3:00 — Close

> AeroXAI is an evidence-aware workflow: detect on real data, explain the warning, test actions in physics, optimize under constraints, explain the recommendation, and refuse to present an unrobust open-loop plan as safe autonomous control.

## Safe numbers to quote

REAL:
- 4/4 documented incidents detected within the timely window.
- 2/4 pre-onset warnings.
- false alert episodes about 0.886 per 24 h.
- PR-AUC about 0.249.

SIMULATED:
- baseline 25.3112 kWh.
- optimized 25.2177 kWh.
- nominal saving 0.0935 kWh / 0.369%.
- high-leak baseline penalty 3.1778 kWh / 12.55%.
- nominal optimizer safety violations: 0.
- 5 of 6 non-nominal stress scenarios unsafe.
- worst tested minimum pressure 3.619 bar(g).

Never combine the 12.55% leak penalty with the 0.369% optimizer saving.

If asked why 0.369% is small:

> The optimizer is constrained by pressure, reserve, startup, minimum-runtime and terminal-pressure requirements. We kept the reproducible result rather than tuning assumptions to create a larger headline number.
