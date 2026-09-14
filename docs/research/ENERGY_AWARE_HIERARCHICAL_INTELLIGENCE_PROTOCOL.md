# AeroXAI Energy-Aware Hierarchical Intelligence Study

Status: **preregister before execution**

Target branch:

research/energy-aware-hierarchical-intelligence

Frozen parent:
8b3b932f321dd7f690ba558420aa13aa2ff3707a

## 1. Research motivation

Industrial AI consumes computational energy while attempting
to reduce physical process energy.

This first study asks whether expensive temporal inference
can be invoked selectively rather than continuously.

The experiment concerns AI inference energy only.

It does not claim measured compressor-energy savings.

## 2. Primary research question

Can frozen Robust-PCA route a causal temporal model only to
suspicious windows while substantially reducing temporal-model
inference compute and retaining coverage of incident-relevant
deep-model evidence?

## 3. Evidence status

MetroPT-3 TEST has already been inspected extensively.

All MetroPT-3 results are therefore:

EXPLORATORY_REUSED_TEST_BENCHMARK

MetroPT2 has also been used in previous research.

It is a cross-dataset transport benchmark, not untouched
confirmation evidence.

## 4. Frozen systems

A. Frozen Robust-PCA operational reference.

B. Always-on causal TCN research reference.

C. Frozen Robust-PCA plus calibration-only escalation gate,
   invoking the same TCN selectively.

System C does not modify the frozen operational alert.

## 5. Frozen PCA reproduction guard

Before interpreting any energy result, reproduce the known
frozen Robust-PCA metrics within 1e-12.

[List the seven frozen values from the YAML.]

Failure invalidates the benchmark.

## 6. Temporal model

Describe exactly the architecture frozen in the YAML:

- causal TCN
- 12 previous 5-minute bins
- same model features
- RobustScaler fit on TRAIN only
- 3 residual blocks
- channels 32/32/32
- kernel 3
- dilation 1/2/4
- dropout 0.10
- next-step feature prediction
- MSE
- AdamW
- lr 1e-3
- weight decay 1e-4
- batch 256
- max 50 epochs
- patience 5
- seed 20260915

No architecture search is permitted.

## 7. Router

Freeze three routing operating points:

q90
q95
q99

Each threshold is estimated only from the official
CALIBRATION PCA causal-EWMA distribution.

No TEST-derived routing threshold is permitted.

## 8. Energy accounting

Primary quantity:
inference compute energy.

Separate:

MEASURED:
direct hardware energy/power measurements.

ESTIMATED:
quantities inferred from software or hardware specifications.

UNKNOWN:
energy that cannot be defensibly measured.

Never silently convert estimated energy into measured energy.

Record:
runtime,
model invocations,
energy where measurable,
latency,
memory,
hardware/software environment.

## 9. Evaluation

Operational detector metrics remain frozen PCA metrics.

For B, report temporal-model anomaly evidence.

For C, report:

- fraction of windows routed to TCN
- coverage of always-on TCN alert bins
- coverage of incident-related TCN alert bins
- coverage of TCN alert episodes
- coverage of incident-related TCN alert episodes
- compute/energy reduction relative to always-on TCN

## 10. Dataset order

1. MetroPT-3 primary benchmark.
2. Freeze result interpretation.
3. MetroPT2 transport benchmark with no retuning.

MetroPT2 architecture, routing quantiles, and definitions
must remain identical.

## 11. Non-goals

This study does not:

- change the production detector;
- use RL;
- use an LLM;
- use RAG;
- modify the digital twin;
- modify the optimizer;
- claim physical plant-energy savings;
- identify physical root cause;
- promote a new detector.

## 12. Decision rule

Advance the hierarchical-compute idea only if:

1. frozen PCA reproduction succeeds;
2. energy/compute measurement is repeatable;
3. selective routing materially reduces TCN computation;
4. incident-relevant TCN evidence is substantially retained;
5. the conclusion holds without TEST-driven retuning.

Otherwise close the branch or revise only through a new
preregistered study.

## 13. Stop condition

After TEST results are observed:

NO router threshold,
TCN architecture,
training hyperparameter,
feature set,
or energy accounting definition may be changed.

A new experiment requires a new protocol commit.