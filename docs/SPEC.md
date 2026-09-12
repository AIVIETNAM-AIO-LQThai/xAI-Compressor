# AeroXAI Technical Specification

Status: FROZEN FOR ESIC 2026 MVP

## 1. Product definition

AeroXAI is an explainable, advisory-first compressed-air
energy decision-support system.

The ESIC MVP combines:

1. real-data compressor anomaly / waste detection;
2. explainable AI for anomaly detection;
3. a physics-based compressed-air digital twin;
4. a baseline compressor controller;
5. constrained energy optimization;
6. explainable operating recommendations;
7. projected energy, cost, and CO2 impact.

The MVP is NOT an autonomous industrial controller.

No command is written to a real PLC or compressor.

---

## 2. Core pipeline

Telemetry
→ Data-quality checks
→ Anomaly / waste detection
→ XAI explanation
→ Physics digital twin
→ Baseline controller
→ Constrained optimizer
→ Action explanation
→ Energy / cost / CO2 estimate

---

## 3. Evidence classes

Every quantitative result exposed by AeroXAI must have exactly
one evidence class.

### REAL

Derived directly from real measured data.

For ESIC MVP:

- MetroPT-3 telemetry;
- anomaly scores;
- incident detection;
- lead time;
- false-alarm statistics.

### SIMULATED

Derived from AeroXAI's physics-based digital twin.

For ESIC MVP:

- compressor scheduling;
- pressure trajectory;
- predicted power;
- baseline energy;
- optimized energy;
- projected savings.

### LITERATURE

Derived from external publications, standards, papers,
government documents, or commercial case studies.

Literature results must never be presented as AeroXAI's
measured performance.

---

## 4. Claims boundary

The ESIC MVP MAY claim:

- detection of anomalous compressor behavior on real
  historical telemetry;
- explainable anomaly scoring;
- physically modeled compressed-air scenarios;
- constrained compressor optimization in simulation;
- projected energy savings under documented simulation
  assumptions.

The ESIC MVP MUST NOT claim:

- verified energy savings in a real factory;
- exact leak localization;
- confirmed physical root cause solely from XAI;
- safe autonomous industrial control;
- universal compatibility with all compressors;
- MetroPT-3 validation of compressor scheduling savings.

---

## 5. Dataset boundary

MetroPT-3 is used ONLY for:

- anomaly detection;
- incident replay;
- explainability;
- event-level evaluation.

MetroPT-3 is NOT used to validate:

- multi-compressor scheduling;
- industrial energy savings;
- exact leak flow;
- factory control safety.

---

## 6. Control boundary

The ESIC MVP operates in:

ADVISORY / SHADOW MODE

The optimization engine produces recommended actions.

A real industrial deployment would require:

1. site calibration;
2. read-only validation;
3. domain-engineer review;
4. supervised operation;
5. safety validation;
6. only then consideration of closed-loop control.

---

## 7. Internal unit convention

AeroXAI uses SI units internally.

Pressure:

- Pa absolute

Temperature:

- K

Mass flow:

- kg/s

Volume:

- m^3

Power:

- W

Energy:

- J internally
- converted to kWh for reporting

Time:

- seconds

Display units may include:

- bar(g)
- °C
- standard m^3/min
- kW
- kWh

Gauge-pressure conversion must explicitly use an atmospheric
reference pressure.

No physics module may silently mix gauge pressure and
absolute pressure.

No physics module may silently mix standard volumetric flow,
actual volumetric flow, and mass flow.

---

## 8. MVP technical priority

Required:

1. MetroPT-3 causal preprocessing
2. statistical anomaly baseline
3. event-level evaluation
4. anomaly explanation
5. physics digital twin
6. rule-based baseline controller
7. constrained optimizer
8. optimizer explanation
9. FastAPI
10. React UI
11. validation and ESIC evidence package

Optional only after all required items work:

- temporal deep learning;
- Integrated Gradients;
- TimeSHAP;
- PPO;
- LLM explanation;
- database;
- live industrial protocol integration.

---

## 9. Scientific rule

Correctness beats model complexity.

A simpler method that survives chronological evaluation and
produces reproducible evidence is preferred over a more
complex model with weaker validation.

---

## 10. Definition of completion

The ESIC MVP is technically complete when it can demonstrate:

1. a real MetroPT incident replay;
2. causal anomaly detection;
3. an explanation of signals associated with the warning;
4. physically sensible compressed-air simulation;
5. a reproducible baseline controller;
6. a lower-energy feasible optimized schedule in at least one
   documented scenario;
7. an explanation of why that schedule was selected;
8. clear REAL / SIMULATED / LITERATURE provenance for every
   quantitative claim.
