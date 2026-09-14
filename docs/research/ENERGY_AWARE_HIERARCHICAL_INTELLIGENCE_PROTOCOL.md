# AeroXAI Energy-Aware Hierarchical Intelligence Study

Status: **pre-TEST protocol amended after TRAIN/CALIBRATION conditioning diagnostic**

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

MetroPT-3 TEST has already been inspected extensively in prior work.

All MetroPT-3 results are therefore:

EXPLORATORY_REUSED_TEST_BENCHMARK

MetroPT2 has also been used in previous research.

It is a cross-dataset transport benchmark, not untouched
confirmation evidence.

For this study, no MetroPT-3 TEST rows were opened during the
G3 TRAIN/CALIBRATION conditioning diagnostic or the scaler amendment.

## 4. Frozen systems

A. Frozen Robust-PCA operational reference.

B. Always-on causal TCN research reference.

C. Frozen Robust-PCA plus calibration-only escalation gate,
   invoking the same TCN selectively.

System C does not modify the frozen operational alert.

## 5. Frozen PCA reproduction guard

Before interpreting any energy result, reproduce the known
frozen Robust-PCA metrics within 1e-12.

The frozen Robust-PCA reference metrics are:

```text
timely incident recall        = 1.0
pre-onset incident recall     = 0.5
incident overlap recall       = 1.0
false alerts / 24h            = 0.8862358575692373
time in alert fraction        = 0.07549570810485644
relevant episode precision    = 0.06338028169014084
PR-AUC                        = 0.24916826514697024
```

Failure invalidates the benchmark.

## 6. Temporal model

Describe exactly the architecture frozen in the YAML after the
pre-TEST scaler amendment:

- causal TCN
- 12 previous 5-minute bins
- same model features
- StandardScaler fit on TRAIN only
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

The frozen operational PCA remains RobustScaler-based and is not
changed by the TCN scaler amendment.

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

## 13. Measurement calibration decision

The GPU energy meter was calibrated before any TCN or MetroPT
energy benchmark.

Calibration configuration:

- NVIDIA GeForce RTX 4090
- in-process NVML sampling
- 100 ms sampling interval
- 2 warm-up workloads
- 5 measured repetitions
- 10 s matched idle baseline
- 20 s synthetic GPU workload
- 2048 x 2048 FP32 matrix multiplication workload

Observed repeatability:

- gross energy CV: 0.274%
- incremental energy CV: 0.217%
- gross energy per iteration CV: 0.388%
- incremental energy per iteration CV: 0.441%

The largest observed calibration CV was 0.441%.

Before observing any TCN or MetroPT benchmark result, the study
therefore freezes a minimum material measured GPU-energy reduction
of 3%.

This threshold is intentionally greater than five times the largest
observed calibration CV:

5 x 0.441% = 2.205%.

A routed system will not be described as materially reducing measured
GPU energy unless its mean measured GPU-energy reduction relative to
the always-on TCN reference is at least 3%.

All individual repetitions, mean values, dispersion, gross GPU energy,
and incremental GPU energy must still be reported.

This threshold applies only to measured GPU energy. CPU energy remains
UNKNOWN under the current WSL hardware interface.

## 14. Pre-TEST temporal-scaling amendment

The first preregistered TCN execution was run on MetroPT-3 TRAIN and
CALIBRATION only. MetroPT-3 TEST was not opened.

That run exposed a numerical-conditioning failure of the originally
frozen RobustScaler + MSE combination.

TRAIN scaled-value diagnostics were:

- absolute p99 approximately 1435
- absolute p99.9 approximately 6434
- absolute maximum approximately 6931

CALIBRATION showed comparable scale magnitudes.

The cause was visible directly in TRAIN feature geometry. Several
features have extremely small TRAIN interquartile ranges relative to
their physically observed range. Examples include:

- `tp2__std`: IQR 0.0007059, raw max 4.8935
- `tp2__last`: IQR 0.0020, raw max 10.544
- `tp2__max`: IQR 0.0040, raw max 10.544
- `dv_pressure__std`: IQR 0.0002156, raw max 1.1759

Under RobustScaler these small denominators create values in the
thousands. With a squared-error objective, a few such coordinates
dominate the temporal-model loss.

The CALIBRATION residual diagnostic confirmed this consequence:

- top-1 feature MSE share: 59.7%
- top-3 feature MSE share: 92.2%
- top-5 feature MSE share: 98.2%

This was treated as a conditioning diagnostic, not as a model-quality
comparison.

Before any MetroPT-3 TEST evaluation, routing benchmark, or TCN energy
comparison, the TCN preprocessing is therefore amended exactly once:

- TCN scaler changes from RobustScaler to StandardScaler;
- StandardScaler is fit on TRAIN only;
- all 63 features remain unchanged;
- TCN architecture remains unchanged;
- MSE objective remains unchanged;
- optimizer and training hyperparameters remain unchanged;
- router operating points remain unchanged;
- frozen Robust-PCA remains unchanged;
- the 3% measured-GPU-energy materiality rule remains unchanged.

No alternative TCN was selected using TEST performance. The amendment
is based on numerical conditioning and TRAIN feature geometry, with
CALIBRATION used only to document that the pathological scaling
propagated beyond TRAIN.

The original RobustScaler TCN output is retained as a diagnostic
artifact and is not the benchmark TCN for Systems B or C.

After this amendment, no further preprocessing, model, or routing
change is permitted before observing the primary TEST result.

## 15. Stop condition

After TEST results are observed:

NO router threshold,
TCN architecture,
training hyperparameter,
feature set,
preprocessing/scaling rule,
or energy accounting definition may be changed.

A new experiment requires a new protocol commit.
