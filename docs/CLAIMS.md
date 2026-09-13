# AeroXAI Claims and Evidence Boundary

Use this file as the ESIC 2026 claims guardrail.

## REAL

Source: MetroPT-3 historical measured telemetry.

Supported:
- anomaly scoring;
- incident detection and timing;
- benchmark alert burden;
- PCA alert-score contribution analysis.

Not supported by MetroPT:
- multi-compressor dispatch savings;
- exact industrial leak flow;
- factory control safety;
- energy-optimization validation.

Approved wording:

> Under the frozen chronological MetroPT-3 benchmark, the robust-scaled PCA detector detected all four documented incidents within the predefined timely window.

> Two of the four documented incidents produced pre-onset warnings; the other two were detected shortly after onset.

> The detector produced about 0.886 benchmark-false alert episodes per 24 hours of valid test exposure and PR-AUC about 0.249.

Do not say AeroXAI predicted all four incidents in advance.

`benchmark_false` means not linked to a documented relevance interval; sparse incident labels mean these episodes are not necessarily physically normal.

### Detection XAI

Approved:

> PCA feature contributions exactly decompose the squared reconstruction-error evidence and are propagated through the same causal EWMA used by the alert pipeline.

Use “signals contributing to anomaly evidence.”

Do not say:
- “PCA identified the root cause.”
- “DV pressure caused the incident.”

## SIMULATED

Source:
- AeroXAI digital twin;
- assumed compressor fleet;
- synthetic demand profile;
- simulated baseline;
- MILP optimizer;
- stress scenarios.

Supported:
- pressure trajectory;
- compressor schedule;
- modeled reserve;
- predicted electrical energy;
- simulated energy differences;
- schedule/constraint explanations;
- robustness outcome.

Not supported:
- measured industrial savings;
- universal safety;
- calibrated plant economics.

### Leak scenario

Approved:

> In the frozen simulated baseline scenario, increasing leak input from 0.005 to 0.020 kg/s increased predicted one-hour electrical energy by about 3.1778 kWh, or 12.55%.

Do not say “AeroXAI saves 12.55%.”

### Optimizer

Approved:

> In the frozen nominal synthetic scenario, constrained MILP reduced predicted electrical energy from 25.3112 to 25.2177 kWh, a reduction of 0.0935 kWh or 0.369%.

Approved:

> The nominal optimized schedule has zero modeled safety violations and satisfies modeled pressure, reserve and terminal-pressure constraints.

The objective includes energy, startup and overpressure penalties. The reported 0.369% refers to electrical energy only.

Do not say every optimized interval uses less energy than baseline.

Do not say MetroPT validates optimizer savings.

### Action explanations

Approved:

> Recommendations are explained using pressure state, safety margin, reserve margin and active/near-active constraints.

These are schedule/constraint explanations, not causal effects of individual actions. Do not call them SHAP.

### Robustness

Approved:

> The nominal schedule is feasible under nominal deterministic assumptions but is not robust to all tested open-loop perturbations.

Approved:

> Five of six non-nominal stress scenarios were unsafe under the simplified receiver model.

Approved:

> The result motivates short-lived advisory actions with mandatory re-optimization rather than execution of the full frozen one-hour schedule.

Frozen safety posture:

```text
deployment_mode = advisory
override_equipment_ctrl = false
recommendation_valid_for_seconds = 60
requires_reoptimization = true
open_loop_schedule_approved = false
```

Do not claim safe autonomous industrial control.

## Cross-evidence rules

Bad:

> MetroPT proves AeroXAI can detect leaks and save 0.369% energy.

Better:

> MetroPT provides real-data evidence for anomaly detection and explainability. Separately, a documented simulated compressor scenario demonstrates constrained energy optimization.

Never combine:
- 12.55% high-leak penalty;
- 0.369% nominal dispatch saving.

## Safe short pitch

> AeroXAI detects and explains anomalous compressed-air behavior on real historical telemetry, then uses a separate physics-based digital twin and constrained optimizer to generate short-lived explainable energy recommendations in simulation, with an explicit robustness gate that prevents unrobust open-loop schedules from being presented as safe control.

Use maturity terms:
- prototype;
- MVP;
- advisory decision-support;
- shadow-mode architecture;
- simulated recommendation layer.

Avoid:
- production-ready autonomous controller;
- guaranteed savings;
- plug-and-play for every compressor.
