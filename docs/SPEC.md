# AeroXAI Technical Specification

Status: **ESIC 2026 V3 RELEASE CANDIDATE**

## 1. Product definition

AeroXAI is a proof-carrying, explainable, advisory-first prototype for compressed-air energy-waste intelligence.

The defining V3 behavior is not simply anomaly detection or compressor optimization. It is the controlled transition between evidence layers:

1. detect anomalous behavior on real telemetry;
2. verify detector evidence;
3. generate candidate waste hypotheses without causal overclaim;
4. check whether a validated physical bridge exists;
5. infer physical state only when the model and observations support it;
6. propagate bounded uncertainty into physical scenarios;
7. optimize one shared action across those scenarios;
8. independently safety-check the first action;
9. issue only a short-lived advisory;
10. carry provenance and claim boundaries with the result.

The MVP is not an autonomous industrial controller and writes no command to a PLC or compressor.

## 2. Implemented V3 architecture

```text
REAL TELEMETRY
MetroPT-3
-> data-quality / chronological preprocessing
-> frozen RobustScaler + PCA detector
-> exact PCA attribution
-> calibration-context rarity
-> past-only temporal evidence
-> model-space counterfactual detector-dependence verification
-> candidate waste hypothesis engine
-> physical-bridge gate
-> WITHHOLD if calibration / required observations are missing

TEMPORAL RESEARCH CHANNEL
5-min causal TCN
causal Transformer
raw-context causal TCN
-> chronological benchmark
-> no production promotion unless evidence beats primary baseline

SIMULATION-ONLY PHYSICS + CONTROL
known synthetic truth
-> synthetic receiver trajectory
-> inverse total-outflow physics
-> identifiability test
-> bounded physical-state uncertainty
-> scenario generation
-> shared-action robust MILP
-> compressor runtime-state constraints
-> independent first-action safety gate
-> 60-second advisory
-> next runtime state
-> re-observe / re-optimize

PROOF LAYER
real evidence replay
+ simulated action proof
-> proof manifest
-> FastAPI endpoint
-> Evidence Proof UI
```

## 3. Provenance model

### REAL

Measured MetroPT-3 telemetry and frozen real-data detector/XAI evaluation.

### MODEL_INFERENCE_FROM_REAL

Candidate hypotheses generated from verified REAL detector evidence.

Requirements:

```text
causal_claim = false
```

These objects are not diagnoses or probabilities of physical root cause.

### PHYSICS_MODEL_INFERENCE

Physical quantities derived from the receiver model.

Requirements:

- assumptions must be explicit;
- identifiability must be explicit;
- values must not be labeled measured unless they are measured;
- causal claim remains false.

### SIMULATED

Synthetic digital-twin, optimizer, safety and energy results.

### LITERATURE

External context only.

## 4. Real-data detector

Primary production baseline:

```text
RobustScaler
-> PCA reconstruction error
-> causal EWMA
-> persistence rule
```

Protocol:

- chronological train/calibration/test split;
- no future leakage;
- threshold from calibration only;
- event-level evaluation;
- frozen detector parameters.

Frozen headline metrics:

```text
timely incident recall      1.0
pre-onset incident recall   0.5
false alerts / 24 h         0.8862358576
PR-AUC                      0.2491682651
```

## 5. Verified detector explanation

Explanation layers:

1. exact PCA reconstruction-error contribution;
2. calibration-only empirical contribution percentile;
3. 12-bin causal temporal evidence window;
4. model-space counterfactual repair;
5. unchanged EWMA / persistence replay;
6. explicit knowledge limit.

Interpretation rule:

> Counterfactual repair verifies detector dependence, not physical causality.

## 6. Temporal representation research

Three causal deep temporal candidates are implemented:

- 5-minute engineered-feature TCN next-state forecaster;
- causal Transformer next-state forecaster;
- raw 10-second context TCN forecasting the next 5-minute engineered target.

All use chronological training and calibration-only thresholds.

None is production-primary.

See `docs/RESEARCH_RESULTS.md` for the comparison and repeated-test-set caveat.

## 7. Waste hypothesis engine

Candidate ontology includes:

- leak-like persistent outflow;
- process-demand surge;
- pressure-regulation instability;
- pressure-sensor inconsistency;
- compressor performance degradation;
- unloaded / inefficient operation;
- excess-pressure operation.

Candidate generation uses verified XAI groups and calibration context.

Output provenance:

```text
evidence_class = MODEL_INFERENCE_FROM_REAL
causal_claim = false
```

## 8. Real physical-bridge gate

A real-asset action is only eligible when the minimum physical bridge is present.

Current required conditions include:

- site-calibrated receiver and compressor model;
- calibrated compressor inflow observation;
- receiver pressure trace.

Current MetroPT prototype:

```text
physical_bridge_available = false
downstream_physics_executed = false
robust_control_executed = false
recommendation = null
```

Current bridge status:

```text
WITHHOLD_REAL_ASSET_ADVISORY_MISSING_VALIDATED_PHYSICAL_BRIDGE
```

This is expected behavior.

## 9. Inverse receiver physics

The implemented inverse model uses:

```text
m_out = m_in - V/(R*T) * dp/dt
```

where `m_out` is total outflow.

Important identifiability rule:

```text
m_out = m_demand + m_leak
```

Pressure plus compressor inflow alone cannot distinguish legitimate process demand from leakage.

If independent demand is absent:

```text
leak_identifiable = false
demand = null
leak = null
```

## 10. Physical-state uncertainty

Uncertainty is represented as bounded interval estimates:

```text
lower
center
upper
```

These are engineering uncertainty bounds, not confidence intervals unless a statistical calibration procedure is separately defined.

The current synthetic proof uses:

```text
total outflow center 0.110 kg/s
absolute bound       +/- 0.004 kg/s
```

which produces approximately:

```text
0.106 / 0.110 / 0.114 kg/s
```

## 11. Scenario generation

For unidentified demand/leak allocation, scenarios vary total outflow only and do not fabricate a demand/leak split.

For identifiable states with independent demand, bounded combinations may be generated subject to physical consistency.

## 12. Robust scheduler

The V3 robust optimizer uses one shared compressor action schedule across all physical scenarios.

Constraints include:

- pressure safety bounds in every scenario;
- modeled reserve in every scenario;
- terminal pressure;
- compressor capacity;
- startup / shutdown transitions;
- minimum on/off runtime;
- VSD load limits;
- persistent runtime state across receding horizons.

Objective includes:

- electrical energy;
- startup penalty;
- overpressure penalty.

Use:

> lowest-objective feasible shared schedule

rather than claiming every action is the instantaneous lowest-energy action.

## 13. First-action safety gate

Only the first optimizer interval is eligible for advisory display.

The first action is independently replayed across the scenario set.

Required output behavior:

```text
robust_safe = true
valid_for_seconds = 60
```

The system then updates compressor runtime state and requires re-observation / re-optimization.

## 14. Simulation-only validation proof

Synthetic truth:

```text
total outflow = 0.110 kg/s
```

Inverse recovery passes with numerical error around machine precision.

Current scenario set:

```text
0.106 / 0.110 / 0.114 kg/s total outflow
```

Leakage remains unidentified.

Current robust first action:

```text
fixed_1 ON
fixed_2 OFF
vsd_1   20%
```

Current first-interval proof includes:

- three source scenarios;
- robust safety pass;
- approximately 6.805 bar(g) worst minimum pressure;
- approximately 6.846 bar(g) worst maximum pressure;
- approximately 0.026 kg/s minimum reserve;
- 60-second validity.

Scope:

```text
SIMULATION_ONLY
connected_to_real_incident = false
```

## 15. Earlier nominal energy benchmark

The earlier frozen one-hour deterministic benchmark remains valid as a separate experiment:

```text
baseline energy            25.3112 kWh
optimized energy           25.2177 kWh
energy reduction            0.0935 kWh
energy reduction percent    0.369%
```

A separate high-leak scenario increased predicted baseline energy by 12.55%.

Those experiments are not the same as the V3 robust-action proof.

## 16. Proof manifest

The product-level evidence manifest combines the paths without scientifically joining them.

Machine-readable file:

```text
docs/proof_manifest.json
```

API:

```text
GET /evidence/proof-manifest
```

UI:

```text
#proof
```

Manifest invariants include:

- real recommendation is null when the bridge is unavailable;
- simulation scope is `SIMULATION_ONLY`;
- simulated proof cannot declare connection to the real incident;
- causal claim is false;
- allowed and forbidden claims are machine-readable.

## 17. Control boundary

```text
deployment_mode = advisory
override_equipment_ctrl = false
recommendation_valid_for_seconds = 60
requires_reoptimization = true
open_loop_schedule_approved = false
```

There is no PLC-write endpoint.

## 18. Unit convention

Internal physics uses explicit SI conventions:

- pressure: Pa absolute internally; bar(g) only when explicitly converted for display;
- temperature: K;
- mass flow: kg/s;
- volume: m3;
- power: kW where documented;
- energy: kWh;
- time: seconds.

Physics modules must not silently mix gauge/absolute pressure or mass/volumetric flow.

## 19. Scientific rules

1. Correctness beats model complexity.
2. Negative results are reported rather than tuned away.
3. A detector explanation is not a physical diagnosis.
4. Identifiability must be stated before estimating latent physical quantities.
5. REAL and SIMULATED provenance must remain separate.
6. A recommendation may only claim the uncertainty set and model constraints actually checked.
7. Repeated use of a held-out test period for research model comparison is disclosed.

## 20. Definition of V3 completion

V3 is technically complete when the repository demonstrates:

1. frozen real MetroPT detector benchmark;
2. verified XAI with detector-dependence testing;
3. causal temporal-representation research with documented non-promotion;
4. candidate waste hypotheses from real detector evidence;
5. explicit physical-bridge eligibility gate;
6. inverse receiver physics with identifiability;
7. bounded physical uncertainty;
8. physical scenario generation;
9. shared-action robust optimization;
10. persistent compressor runtime state;
11. independent first-action safety replay;
12. 60-second advisory behavior;
13. real/simulated proof artifacts;
14. machine-readable proof manifest;
15. FastAPI endpoint and React Evidence Proof UI;
16. passing backend tests, Ruff and production frontend build;
17. successful browser smoke test.

The current V3 release branch should be frozen after final acceptance and documentation review.
