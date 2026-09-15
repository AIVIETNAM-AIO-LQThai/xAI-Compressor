# CoBra Router V2 Final Review

Status: **completed external evaluation — closed**

Study: `cobra_router_v2_adaptation`

Final main commit containing the frozen result: `82e06ed6fbd6f729f70ddba0650f2917c0b99ff3`

## Purpose

This review closes the CoBra Router V2 study and records the full chronology,
frozen decision rules, one-shot future evaluation, measured GPU-energy result,
and claim boundaries.

The study asked whether the frozen evidence-aware Router V2 could preserve
high-value temporal-model evidence while reducing expensive TCN inference on a
new industrial compressor-related telemetry domain.

The CoBra result is **not zero-shot**. CoBra used a preregistered within-domain
TRAIN/CAL adaptation followed by one-shot evaluation on the chronologically
later EVAL partition.

## Dataset role

Dataset:

> CoBra High Temperature Heat Pump Demonstrator — Experimental Dataset 2024–2025

The source was locked before model development on CoBra sensor values.

Frozen whole-day chronology:

```text
TRAIN  first 14 days
CAL    next 4 days
EVAL   final 5 days
```

EVAL days:

```text
2025-02-17
2025-02-18
2025-02-20
2025-02-24
2025-02-25
```

The EVAL partition was not used for feature selection, model fitting,
threshold selection, router redesign, or post-hoc rescue.

## Frozen feature and temporal contract

Before CoBra sensor outcomes were opened, the adaptation protocol froze:

```text
257 structural candidate fields
→ metadata-only exclusion rule
→ 219 final raw features
```

Final feature-set SHA256:

```text
ba54ffa70cda5690ceaef4d0a58d6d14129dd36a0e03abbc95c20344e3e4a9aa
```

Representation:

```text
5-minute bins
arithmetic mean per raw channel
no interpolation
no forward/back fill
no synthetic seconds
no cross-day windows
no gap bridging
invalid bin if any frozen raw feature lacks a finite observation
```

The temporal model used 12 prior 5-minute bins to predict the next bin.

## Frozen Router V2 policy

Router V2 was frozen before CoBra model outcomes.

Online router inputs:

```text
pca_ewma_percentile
recent_q90_hit_fraction
recent_q95_hit_fraction
```

Frozen router family:

```text
max3
```

Frozen threshold:

```text
0.40
```

A valid TCN target with unavailable router inputs routes to TCN.
An invalid temporal target is not an eligible evaluation unit.

No TCN score or incident label is used as an online routing input.

## TRAIN adaptation

TRAIN representation:

```text
source rows          310,869
valid 5-minute bins      679
```

PCA:

```text
RobustScaler
variance retained target >= 0.95
fitted components = 1
explained variance ≈ 0.995816
```

TCN:

```text
eligible sequences       475
fit samples              380
validation samples        95
device                   CUDA
best epoch                48
best validation MSE       0.34722900390625
```

The one-component PCA result was accepted as the direct consequence of the
frozen variance-retention rule. It was not altered after inspection.

## CALIBRATION adaptation

CAL representation:

```text
valid 5-minute bins      246
contiguous segments        5
eligible TCN targets     194
```

Frozen CAL thresholds:

```text
PCA EWMA q90   88.89060240044532
PCA EWMA q95   94.27901488205916

TCN q90         0.6537802815437317
TCN q99.5       2.2471232414245605
```

CAL teacher evidence:

```text
high-evidence units       20
alert-evidence units       1
```

Frozen Router V2 CAL behavior:

```text
TCN invocation fraction    0.4948453608
high-evidence coverage     1.0000000000
alert coverage             1.0000000000
```

The CAL alert count was below the preregistered adequacy minimum of three, so
the alert coverage value is descriptive only.

## One-shot EVAL result

The frozen implementation was committed before the five EVAL archives were
opened.

EVAL representation:

```text
source rows              125,826
valid 5-minute bins          358
invalid 5-minute bins         68
contiguous segments           11
eligible TCN targets         247
```

Frozen teacher evidence:

```text
high-evidence units           93
alert-evidence units           1
```

Router V2:

```text
TCN invocations              194 / 247
TCN invocation fraction       0.7854251012
TCN call reduction            0.2145748988
high-evidence coverage        1.0000000000
alert-evidence coverage       1.0000000000
forced-missing routes         0
```

Preregistered evidence criteria:

| Criterion | Requirement | Observed | Status |
|---|---:|---:|---|
| High-evidence unit count | >= 20 | 93 | Adequate |
| High-evidence coverage | >= 0.80 | 1.00 | **PASS** |
| Alert-evidence unit count | >= 3 | 1 | **INSUFFICIENT_EVIDENCE** |
| Alert coverage | >= 0.80 when adequate | 1.00 descriptive | Not evaluated |

The correct interpretation is:

> Router V2 retained all 93 high-evidence TCN units on the future CoBra EVAL
> partition after preregistered within-domain adaptation.

It is not valid to claim alert-level transport because only one alert-evidence
unit occurred.

## GPU-energy measurement

Measurement scope:

```text
GPU device
TCN scoring only
NVIDIA GeForce RTX 4090
```

Not measured:

```text
CPU/router energy
whole-system energy
physical compressor energy
plant energy savings
carbon savings
```

Frozen protocol:

```text
NVML interval                 0.1 s
warmups                       2 per condition
measured repetitions          5
matched idle                 10 s
minimum workload             20 s
always-on invocations       247 / logical pass
Router V2 invocations       194 / logical pass
```

Measured gross GPU-energy reduction versus always-on TCN:

```text
rep 1   0.0988968409
rep 2   0.0839119404
rep 3   0.0658414005
rep 4   0.1197916969
rep 5   0.0992418922
```

Summary:

```text
mean gross reduction          0.0935367542
                              = 9.3537%

positive reduction            5 / 5 repetitions
preregistered minimum mean   50%
```

GPU-energy criterion:

```text
FAIL
```

The positive direction was consistent, but the magnitude was far below the
preregistered 50% threshold.

## Final preregistered verdict

| Component | Final status |
|---|---|
| High-evidence coverage | **PASS** |
| Alert-evidence coverage | **INSUFFICIENT_EVIDENCE** |
| GPU-energy reduction | **FAIL** |
| Overall preregistered status | **FAIL** |

The overall result is therefore a genuine negative result under the frozen
joint success rule.

## Interpretation

The CoBra study supports a narrower but useful conclusion:

> After preregistered within-domain adaptation, evidence-aware Router V2
> transported its high-evidence selection behavior to the future CoBra
> partition, retaining 100% of 93 adequate high-evidence TCN units while
> skipping 21.46% of eligible TCN calls. However, measured gross GPU-energy
> reduction averaged only 9.35%, so the preregistered 50% compute-energy target
> failed.

The difference between call reduction and measured energy reduction is itself
important. GPU energy is not expected to scale one-for-one with logical TCN
invocation count because batching, fixed device overhead, occupancy, memory
traffic, and baseline power can dominate small inference workloads.

This study does not justify changing the frozen routing threshold after EVAL.

## Claim boundaries

Supported:

- the CoBra EVAL partition was chronological and later than TRAIN/CAL;
- the final policy was frozen before EVAL;
- high-evidence coverage was 100% on 93 adequate EVAL units;
- TCN calls were reduced by 21.46%;
- gross GPU-energy reduction was positive in all five measured repetitions;
- mean measured gross GPU-energy reduction was 9.35%.

Not supported:

- zero-shot generalization;
- universal router generalization;
- alert-level transport on CoBra;
- >=50% GPU-energy savings on CoBra;
- CPU-energy savings;
- whole-system computing-energy savings;
- physical compressor-energy savings;
- plant energy savings;
- carbon savings;
- causal fault diagnosis;
- autonomous equipment control.

## Closure decision

The study is closed.

No post-EVAL rescue, threshold tuning, feature redesign, model refit, selective
re-reporting, or repeated independent claim is allowed on this CoBra EVAL
partition.

Future router work requires a new research question and new independent
validation data. The current CoBra EVAL partition must be treated as previously
seen evaluation data in any later study.
