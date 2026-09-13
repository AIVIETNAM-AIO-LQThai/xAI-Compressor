# AeroXAI Technical Specification

Status: **FROZEN FOR ESIC 2026 MVP**

## 1. Product definition

AeroXAI is an explainable, advisory-first compressed-air energy decision-support system.

The implemented MVP combines:
1. real-data anomaly / waste detection;
2. explainable anomaly scoring;
3. a physics-based compressed-air digital twin;
4. a simulated conventional baseline controller;
5. constrained energy optimization;
6. explainable operating recommendations;
7. robustness stress testing and an advisory safety gate;
8. FastAPI and React interfaces.

The MVP is not an autonomous industrial controller and writes no command to a PLC or compressor.

## 2. Implemented pipeline

```text
REAL TELEMETRY
Telemetry
-> data-quality checks
-> chronological feature generation
-> anomaly detection
-> PCA alert explanation

SIMULATED DECISION LAYER
scenario / forecast inputs
-> physics digital twin
-> baseline controller
-> constrained optimizer
-> action explanation
-> robustness / safety gate
-> predicted energy impact

APPLICATION
FastAPI -> React operator interface
```

Cost and CO2 conversion are not required for the frozen technical MVP and are not presented as implemented evidence.

## 3. Evidence classes

### REAL
- MetroPT-3 telemetry;
- anomaly scores;
- incident detection and timing;
- false-alert benchmark statistics;
- PCA anomaly-score contributions.

### SIMULATED
- compressor scheduling;
- pressure trajectory;
- predicted power / electrical energy;
- baseline and optimized energy;
- action explanations;
- robustness stress tests.

### LITERATURE
External context only. Literature values must never be presented as AeroXAI measured or simulated performance.

## 4. Claims boundary

May claim:
- anomaly detection on real historical telemetry;
- explainable anomaly scoring;
- physically modeled compressed-air scenarios;
- constrained optimization in simulation;
- simulated energy differences under documented assumptions;
- short-lived advisory recommendations with explicit robustness status.

Must not claim:
- verified real-factory savings;
- exact leak localization;
- confirmed root cause from XAI;
- safe autonomous industrial control;
- universal compressor compatibility;
- MetroPT validation of scheduling savings.

## 5. Dataset boundary

MetroPT-3 is used only for detection, incident replay, explainability and event-level evaluation.

It is not used to validate scheduling, industrial energy savings, exact leak flow or control safety.

## 6. Control boundary

```text
deployment_mode = advisory
override_equipment_ctrl = false
recommendation_valid_for_seconds = 60
requires_reoptimization = true
open_loop_schedule_approved = false
```

The full optimizer horizon is a planning forecast, not an approved open-loop command sequence.

A real deployment would require site calibration, read-only validation, domain-engineer review, supervised operation and formal uncertainty/safety validation before any closed-loop consideration.

## 7. Unit convention

Internal physics uses explicit SI conventions:
- pressure: Pa absolute;
- temperature: K;
- mass flow: kg/s;
- volume: m3;
- power: W/kW as explicitly stated;
- energy: reported in kWh;
- time: seconds.

Gauge pressure conversion must use an atmospheric reference. Physics modules must not silently mix gauge/absolute pressure or mass/volumetric flow.

## 8. Frozen technical choices

Detection:
- chronological split;
- RobustScaler + PCA reconstruction error primary;
- EWMA smoothing and persistence;
- event-level evaluation;
- exact PCA residual contributions.

Simulation/control:
- lumped isothermal ideal-gas receiver;
- documented assumed compressor fleet;
- pressure-band simulated baseline;
- MILP scheduling;
- pressure, reserve, startup, minimum-runtime and terminal-pressure constraints.

Safety:
- open-loop replay stress testing;
- advisory-only UI/API;
- 60-second recommendation validity;
- mandatory re-optimization.

## 9. Scientific rule

Correctness beats model complexity.

Negative robustness results are reported and used to change product behavior rather than tuned away.

## 10. Definition of completion

The MVP is technically complete when it demonstrates:
1. real MetroPT incident replay;
2. chronological detection without future leakage;
3. alert-signal explanation;
4. physically sensible simulation;
5. reproducible baseline control;
6. a lower-energy feasible nominal optimized schedule;
7. current-action explanation;
8. open-loop robustness testing;
9. safety behavior responding to robustness failure;
10. explicit REAL / SIMULATED / LITERATURE provenance;
11. a functioning FastAPI + React demonstration interface.

The frozen repository implements these requirements.
