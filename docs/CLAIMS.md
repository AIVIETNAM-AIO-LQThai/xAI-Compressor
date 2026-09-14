# AeroXAI Claims and Evidence Boundary

Use this file as the ESIC 2026 claims guardrail.

## Core rule

AeroXAI must never convert an evidence label into a stronger claim than the underlying source supports.

The V3 product rule is:

> **LLM or model components may propose and communicate; data, physics and optimization determine what is verified.**

All current V3 reasoning outputs use `causal_claim = false`.

## Provenance labels

### REAL

Measured MetroPT-3 historical telemetry and frozen evaluation results.

Supported:

- anomaly scores;
- documented-incident detection and timing;
- benchmark alert burden;
- exact PCA reconstruction-error contribution analysis;
- calibration-context contribution percentiles;
- past-only temporal contribution evidence;
- model-space counterfactual detector-dependence verification.

Not supported by REAL MetroPT evidence:

- exact leak mass flow;
- multi-compressor dispatch savings;
- factory control safety;
- measured optimization savings;
- confirmed physical root cause.

Approved wording:

> Under the frozen chronological MetroPT-3 benchmark, the robust-scaled PCA detector detected all four documented incidents within the predefined timely window.

> Two of the four documented incidents produced pre-onset warnings; the other two were detected shortly after onset.

> The detector produced about 0.886 benchmark-false alert episodes per 24 hours of valid test exposure and PR-AUC about 0.249.

Do not say AeroXAI predicted all four documented incidents in advance.

`benchmark_false` means not linked to a documented relevance interval. Sparse incident labels mean such episodes are not necessarily physically normal.

### MODEL_INFERENCE_FROM_REAL

Candidate hypotheses generated from verified detector evidence.

Examples in the current incident replay include:

- leak-like persistent outflow;
- process-demand surge;
- sensor inconsistency;
- compressor performance degradation;
- pressure-regulation instability.

Approved wording:

> AeroXAI converts verified anomaly evidence into ranked candidate waste hypotheses while preserving alternatives and explicitly avoiding a causal diagnosis.

Do not say:

- “The detector diagnosed a leak.”
- “The highest-ranked hypothesis is the true fault.”
- “A candidate hypothesis is a probability of root cause.”

### PHYSICS_MODEL_INFERENCE

Quantities inferred through the receiver model.

Approved wording:

> Receiver dynamics can constrain total outflow under documented assumptions.

Critical identifiability rule:

```text
total outflow = legitimate process demand + leakage
```

Without an independent process-demand observation, demand and leakage remain observationally equivalent.

Therefore do not claim leakage has been identified when `leak_identifiable = false`.

### SIMULATED

Source:

- synthetic receiver dynamics;
- assumed compressor fleet;
- bounded physical scenarios;
- digital twin;
- robust optimizer;
- safety replay;
- earlier nominal baseline/optimizer experiments.

Supported:

- synthetic pressure trajectory;
- synthetic inverse-physics recovery;
- bounded physical-state scenarios;
- compressor schedule / first action;
- modeled reserve;
- predicted electrical energy;
- simulated energy differences;
- modeled safety-gate outcome.

Not supported:

- measured industrial savings;
- universal safety;
- calibrated plant economics;
- any statement that a simulated physical state describes a real MetroPT incident.

### LITERATURE

External context only.

Literature values must never be presented as AeroXAI measured or simulated performance.

## Detection XAI

Approved:

> PCA feature contributions exactly decompose squared reconstruction-error evidence and are propagated through the same causal EWMA used by the alert pipeline.

Approved:

> Contribution percentiles compare each signal group's anomaly contribution against its calibration-only reference distribution. They measure unusual detector evidence, not fault probability.

Approved:

> Across all four frozen incident explanations, the dominant contribution group was at or above approximately the 99.55th calibration-context percentile.

Approved:

> Under the documented PCA feature-space counterfactual protocol, repairing the dominant contribution group's anomalous model-space evidence and rerunning the unchanged EWMA and persistence pipeline cleared the frozen alert in all four documented incidents.

Interpret this as **detector dependence**, not physical causality.

Do not say:

- “The counterfactual proves the physical root cause.”
- “Repairing TP2 or DV pressure in the factory will remove the fault.”
- “100th percentile means 100% probability of being the fault.”
- “The temporal heatmap identifies physical fault onset.”
- “PCA identified the root cause.”

Use:

> signals contributing to anomaly evidence

## Temporal-model research claims

The frozen PCA detector remains the primary production baseline.

Approved:

> Causal TCN and Transformer models were benchmarked under chronological train/calibration/test rules but did not outperform aligned PCA on alert burden and ranking quality.

Approved:

> A raw-context causal TCN produced pre-onset alerts for all four documented incidents on its valid target subset, but at substantially higher false-alert burden than aligned PCA, so it was retained only as an exploratory precursor channel.

Required caveat:

> Multiple temporal-model experiments reused the held-out test period for research comparison. Therefore the raw-context precursor finding is exploratory and is not an independent confirmatory estimate of future generalization.

Do not say:

- “The raw TCN predicts all leaks.”
- “The raw TCN is better than PCA.”
- “100% pre-onset recall proves production readiness.”

## Real evidence-to-action boundary

For the current MetroPT prototype the proof manifest records:

```text
physical_bridge_available = false
downstream_physics_executed = false
robust_control_executed = false
recommendation = null
```

Missing requirements include:

- site-calibrated receiver and compressor model;
- calibrated compressor inflow observation;
- receiver pressure trace appropriate for inverse physics.

Approved:

> AeroXAI intentionally withholds real-asset physical reasoning and control advice when the physical bridge is not validated.

This is a core safety and scientific-integrity claim.

Do not describe the simulation-only action as the action for the real MetroPT incident.

## Simulation-only robust proof

Synthetic truth:

```text
total outflow = 0.110 kg/s
```

Inverse-physics recovery:

```text
mean inferred total outflow = 0.11000000000000001 kg/s
absolute numerical error    ≈ 1.39e-17 kg/s
```

Bounded total-outflow scenarios:

```text
approximately 0.106 / 0.110 / 0.114 kg/s
```

Leakage remains unidentified because independent process demand is intentionally absent.

Approved:

> In the simulation-only proof, AeroXAI recovered known synthetic total outflow, propagated bounded uncertainty into three physical scenarios, optimized one shared compressor action across all scenarios, and independently verified the first 60-second action against modeled pressure and reserve constraints.

Current simulated action:

```text
fixed_1 ON
fixed_2 OFF
vsd_1   20%
```

Approved:

> The selected simulated first action remained within the modeled safety envelope for all three bounded scenarios in the independent first-action replay.

Do not say:

- “This is the correct action for the MetroPT incident.”
- “This proves safe industrial autonomous control.”
- “The 6.805–6.846 bar(g) range means the controller maintains exactly 7 bar.”

Use:

> remained within the modeled safety envelope

## Earlier frozen energy benchmarks

### High-leak scenario

Approved:

> In the frozen simulated baseline scenario, increasing leak input from 0.005 to 0.020 kg/s increased predicted one-hour electrical energy by about 3.1778 kWh, or 12.55%.

Do not say “AeroXAI saves 12.55%.”

### Nominal optimizer

Approved:

> In the frozen nominal synthetic scenario, constrained MILP reduced predicted electrical energy from 25.3112 to 25.2177 kWh, a reduction of 0.0935 kWh or 0.369%.

The objective includes energy, startup and overpressure penalties. The 0.369% number refers to electrical energy only.

Do not say MetroPT validates optimizer savings.

### Open-loop robustness

Approved:

> The frozen nominal one-hour schedule was feasible under nominal assumptions but was not robust to all tested open-loop perturbations.

Approved:

> That negative result motivated the V3 short-lived, re-optimized robust first-action architecture.

Frozen deployment posture:

```text
deployment_mode = advisory
override_equipment_ctrl = false
recommendation_valid_for_seconds = 60
requires_reoptimization = true
open_loop_schedule_approved = false
```

## Cross-evidence rules

Bad:

> MetroPT proves AeroXAI detects leaks and saves energy with this compressor command.

Better:

> MetroPT provides real-data evidence for anomaly detection, verified detector explanation and candidate waste hypotheses. Because the real physical bridge is not calibrated, AeroXAI withholds a real action. Separately, a synthetic physics proof validates inverse-state reasoning, uncertainty propagation and robust-safe simulated advisories.

Never combine:

- real MetroPT detector evidence with synthetic kg/s values as though they were one measurement chain;
- the 12.55% high-leak penalty with the 0.369% nominal dispatch saving;
- simulated safety with real-factory safety.

## Safe short pitch

> AeroXAI is a proof-carrying compressed-air energy copilot. On real historical telemetry it detects anomalous behavior, verifies why the detector raised the warning, and turns that evidence into non-causal candidate waste hypotheses. Before issuing any real-asset action, it checks whether the physical bridge is calibrated; in the current MetroPT prototype it refuses to cross that boundary. Separately, a physics-based simulation validates inverse-state estimation, bounded uncertainty and robust 60-second compressor advisories. Every recommendation carries its evidence provenance and claim limits.

Use maturity terms:

- prototype;
- MVP;
- advisory decision-support;
- shadow-mode architecture;
- simulation-only action proof;
- proof-carrying recommendation architecture.

Avoid:

- production-ready autonomous controller;
- guaranteed savings;
- confirmed leak diagnosis;
- plug-and-play for every compressor.
