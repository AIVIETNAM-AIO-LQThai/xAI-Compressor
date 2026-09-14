# AeroXAI Temporal Representation Research Results

This file summarizes the causal temporal-model experiments added during V3.

## Research question

Can a deeper temporal representation improve early anomaly sensitivity over the frozen PCA detector without creating an unacceptable alert burden?

The answer so far is:

> **No model earned promotion over PCA.**

The experiments did reveal a potentially useful secondary precursor signal in raw high-frequency context, but it remains exploratory.

## Protocol shared by the temporal models

Where applicable:

- chronological train / calibration / test split;
- model fit only on training data;
- train-only robust scaling;
- internal tail of training used for early stopping;
- official calibration partition used for threshold selection;
- causal EWMA with alpha 0.2;
- calibration quantile 0.995;
- persistence 3 hits in 4 bins;
- incident-level evaluation;
- `evidence_class = REAL`;
- `causal_claim = false`.

## Frozen PCA reference

Full frozen PCA benchmark:

| Metric | PCA |
|---|---:|
| Timely incident recall | 1.000 |
| Pre-onset incident recall | 0.500 |
| Incident overlap recall | 1.000 |
| False alerts / 24 h | 0.886 |
| Time in alert | 0.0755 |
| PR-AUC | 0.249 |
| Relevant episode precision | 0.0634 |

This remains the production baseline.

## 5-minute engineered-feature TCN

Input:

```text
12 causal 5-minute bins
x 63 engineered features
```

Task:

```text
forecast next 63-feature state
-> MSE forecast residual
-> anomaly score
```

TCN result:

| Metric | TCN | Aligned PCA |
|---|---:|---:|
| Timely incident recall | 1.000 | 1.000 |
| Pre-onset incident recall | 0.750 | 0.500 |
| False alerts / 24 h | 2.009 | 0.639 |
| Time in alert | 0.1566 | 0.0747 |
| PR-AUC | 0.0956 | 0.2586 |
| Relevant episode precision | 0.0374 | 0.1000 |

Interpretation:

- TCN increased pre-onset recall by 0.25;
- alert burden increased substantially;
- PR-AUC fell substantially;
- PCA remained superior as the primary detector.

Status:

```text
NOT PROMOTED
```

## Causal Transformer

Architecture:

```text
d_model = 128
nhead = 4
layers = 3
feedforward = 256
dropout = 0.1
```

Transformer result:

| Metric | Transformer | Aligned PCA |
|---|---:|---:|
| Timely incident recall | 1.000 | 1.000 |
| Pre-onset incident recall | 0.500 | 0.500 |
| False alerts / 24 h | 1.753 | 0.639 |
| Time in alert | 0.1119 | 0.0747 |
| PR-AUC | 0.1648 | 0.2586 |
| Relevant episode precision | 0.0276 | 0.1000 |

Relative to the TCN, the Transformer reduced false alerts and improved PR-AUC, but lost the TCN's extra pre-onset recall.

Status:

```text
NOT PROMOTED
```

## Raw-context causal TCN

Research question:

> Does raw 10-second causal context provide earlier precursor information than 5-minute engineered sequences?

Input:

```text
15 raw telemetry sensors
10-second causal resampling
60-minute context = 360 samples
5-minute forecast gap
target = next 63-feature engineered 5-minute state
```

No interpolation or forward filling is used. Missing raw intervals remain gaps.

Raw-context result:

| Metric | Raw-context TCN | Aligned PCA |
|---|---:|---:|
| Timely incident recall | 1.000 | 1.000 |
| Pre-onset incident recall | **1.000** | 0.500 |
| Incident overlap recall | 1.000 | 1.000 |
| False alerts / 24 h | 2.579 | 0.679 |
| Time in alert | 0.1226 | 0.0591 |
| PR-AUC | 0.2331 | 0.4430 |
| Relevant episode precision | 0.0284 | 0.1000 |

First alerts relative to documented onset:

| Incident | Raw-context TCN |
|---|---:|
| 1 | 18.08 h pre-onset |
| 2 | 15.75 h pre-onset |
| 3 | 2.33 h pre-onset |
| 4 | 14.17 h pre-onset |

Interpretation:

- all four documented incidents received pre-onset alerts on the valid raw-context test subset;
- false-alert burden was about 3.8x the aligned PCA rate;
- PR-AUC was approximately half the aligned PCA PR-AUC on that subset;
- therefore the raw-context TCN is not a production replacement for PCA.

Status:

```text
EXPLORATORY SECONDARY PRECURSOR CHANNEL
```

## Why aligned PCA metrics differ between reports

The raw-context model requires a contiguous high-frequency history and a forecast gap, so it has fewer valid target timestamps than the 5-minute sequence experiments.

Therefore:

- `docs/tcn_detector_benchmark.json` and `docs/transformer_detector_benchmark.json` use one aligned PCA subset;
- `docs/raw_context_tcn_benchmark.json` uses a different aligned PCA subset;
- the full frozen PCA benchmark uses the complete frozen valid test exposure.

Only compare a temporal model to the PCA metrics aligned to that report.

## Test-set reuse caveat

The temporal models were explored sequentially while reusing the same held-out test period for comparative research.

Consequences:

- model-selection decisions were informed by prior test outcomes;
- the raw-context TCN's 4/4 pre-onset result should not be treated as an untouched final generalization estimate;
- future confirmation requires a genuinely unseen asset/time period or pre-registered external validation.

Approved wording:

> Raw high-frequency context showed promising precursor sensitivity in exploratory retrospective testing, but did not beat PCA on overall ranking quality or alert burden and requires independent validation.

## Production decision

Primary detector:

```text
RobustScaler + PCA reconstruction error
```

Secondary research channel:

```text
raw-context causal TCN
```

Not promoted:

```text
5-minute TCN
causal Transformer
```

Promotion criteria for future work should include:

- independent validation data;
- comparable or lower false-alert burden than PCA;
- improved PR-AUC or clearly justified operating-point benefit;
- stable behavior across assets/regimes;
- preserved causal/chronological evaluation;
- explainability appropriate to deployment risk.
