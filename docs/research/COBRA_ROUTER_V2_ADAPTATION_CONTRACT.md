# CoBra Router V2 Adaptation Contract

Status: **FREEZE BEFORE COBRA SENSOR VALUES**

Router V2 policy parent:

`392c54f4bae7be632c89ef677b09aa44c09dfec5`

The selected Router V2 policy is already closed and cannot change because of
CoBra behavior.

## Purpose

Freeze the final CoBra feature-selection rule, numeric parsing semantics,
temporal representation, PCA adaptation, TCN architecture/training procedure,
CALIBRATION references, Router V2 mapping, gap behavior, and evaluation
firewall before CoBra TRAIN/CAL model outcomes are opened.

CoBra retains the role:

`FUTURE_GENUINELY_INDEPENDENT_EXTERNAL_VALIDATION`

This is not a zero-shot test. TRAIN and CALIBRATION adaptation are permitted
only under this frozen contract.

## Raw feature selection

The only starting set is the already-frozen 257-field structural compatibility
envelope.

No CoBra sensor value or distribution may influence raw-feature selection.

Fields are removed only by deterministic metadata-name rules.

Exclude exact fields:

- `Day`
- `Time`
- `Operating_status_PLC`

Also exclude a field when its case-folded exact source name contains any of:

- `target`
- `is_on`
- `is_open`
- `is_closed`
- `energy_consumption`
- `energy_output`
- `apparent_energy`
- `reactive_energy`

The purpose is to exclude metadata, commands/setpoints, discrete operating
state declarations, and cumulative energy counters while retaining physical
measurement channels such as flows, pressures, temperatures, vibration,
speed, level, electrical quantities, torque, power, and measured positions.

This is a metadata rule, not value-based feature selection.

The resulting exact feature names must be sorted lexicographically and hashed
before any CoBra sensor values are opened.

No feature may later be removed because of TRAIN, CALIBRATION, or EVALUATION
performance.

## Numeric parsing

Selected raw fields are converted numerically.

Blank, malformed, nonnumeric, NaN, or infinite samples are treated as missing.

No interpolation, forward fill, backward fill, or learned imputation is
allowed.

A valid 5-minute model bin requires at least one finite observed sample for
every frozen raw feature.

If that condition is not satisfied, that bin is invalid rather than repaired.

## Five-minute representation

All 23 source days use the same representation.

Published source timestamps are floored to their corresponding five-minute
boundary.

No artificial timestamp is generated.

Within each source day and five-minute bin, each frozen raw channel is
represented by the arithmetic mean of its finite observed samples.

No synthesized empty bins are inserted.

The three minute-resolution source days remain valid under this rule because
repeated published minute timestamps are aggregated directly. No within-minute
physical elapsed time is inferred.

Source row order remains preserved in ingestion, but row order is not treated
as evidence of one-second timing.

## Temporal contiguity

A valid temporal segment consists only of successive five-minute model bins.

Any step other than exactly five minutes breaks the segment.

No sequence crosses:

- a genuine chronology gap;
- an invalid model bin;
- a frozen source-day boundary.

No gap bridging or interpolation is allowed.

## PCA evidence

Robust-PCA is fit using TRAIN only.

Frozen PCA design:

- RobustScaler;
- 95% retained variance;
- same frozen model-feature representation as the TCN.

PCA reconstruction error is converted into a causal EWMA with alpha `0.20`.

EWMA state resets:

- at every source-day boundary;
- after any break in exact five-minute contiguity.

## Temporal evidence model

The temporal model reuses the already-established causal TCN architecture.

Representation:

- five-minute bins;
- 12 history bins;
- 60-minute causal history;
- one-bin-ahead prediction target;
- exact five-minute contiguity.

Architecture:

- 3 residual blocks;
- channels `[32, 32, 32]`;
- kernel size `3`;
- dilations `[1, 2, 4]`;
- dropout `0.10`;
- output dimension equal to input feature dimension.

Training:

- StandardScaler fit on TRAIN only;
- next-step mean-squared prediction error;
- AdamW;
- learning rate `0.001`;
- weight decay `0.0001`;
- batch size `256`;
- maximum 50 epochs;
- early-stopping patience `5`;
- seed `20260915`;
- chronological 80/20 TRAIN sequence split;
- no CALIBRATION training;
- no EVALUATION training;
- no hyperparameter search.

CoBra performance cannot alter these choices.

## CALIBRATION-only quantities

After TRAIN adaptation is frozen, CALIBRATION may derive only the quantities
declared here.

TCN teacher:

- high evidence = TCN score >= CAL q0.90;
- alert evidence = TCN score >= CAL q0.995;
- quantile interpolation = `higher`.

PCA router references:

- empirical reference distribution = all eligible CAL PCA-EWMA bins;
- PCA q90 = CAL PCA-EWMA q0.90;
- PCA q95 = CAL PCA-EWMA q0.95;
- quantile interpolation = `higher`.

No EVALUATION information may alter any reference or threshold.

## Frozen Router V2 mapping

Router V2 is fixed to:

`max3 >= 0.40`

Inputs:

1. `pca_ewma_percentile`
2. `recent_q90_hit_fraction`
3. `recent_q95_hit_fraction`

`pca_ewma_percentile` is the right-inclusive empirical percentile of the
current causal PCA-EWMA value against the frozen CAL reference distribution.

`recent_q90_hit_fraction` is the fraction of the current and previous
11 contiguous five-minute bins whose PCA-EWMA value is at or above the frozen
CAL q90.

`recent_q95_hit_fraction` is defined identically using frozen CAL q95.

Router score:

`max(input_1, input_2, input_3)`

Route to the TCN when:

`score >= 0.40`

If the TCN target is valid but any Router V2 input is unavailable, route to the
TCN conservatively.

If the TCN target itself is invalid because the required causal temporal
context is unavailable, that target is excluded from eligible routing units.

The Router V2 mathematical form and threshold cannot change after CoBra is
opened.

## External success criteria

On the one-shot five-day EVALUATION:

- high-evidence coverage >= 80%, when at least 20 eligible high-evidence units
  exist;
- alert-evidence coverage >= 80%, when at least 3 eligible alert-evidence
  units exist.

If support is below the required count, report
`INSUFFICIENT_EVALUATION_SUPPORT`.

The compute-energy criterion remains:

- mean gross GPU TCN-scoring energy reduction >= 50% versus always-on TCN;
- positive gross reduction in all five repetitions.

The energy claim is GPU-device TCN scoring only.

It is not plant energy, whole-system compute energy, or carbon savings.

## Evaluation firewall

Before EVALUATION is opened, freeze and hash:

- the exact raw-feature list;
- five-minute preprocessing implementation;
- TRAIN scaler/PCA artifacts;
- TCN checkpoint;
- CAL PCA reference distribution;
- CAL PCA q90/q95;
- CAL TCN q90/q995;
- Router V2 policy;
- evaluation implementation;
- energy-measurement implementation.

During or after EVALUATION there is no:

- fitting;
- threshold selection;
- feature selection;
- architecture change;
- temporal-window change;
- imputation redesign;
- fallback redesign;
- rescue tuning presented as independent validation.

## Claim boundary

A successful EVALUATION may support transport of the frozen hierarchical
routing approach to the externally selected CoBra controlled
heat-pump/compressor domain after preregistered TRAIN/CAL adaptation.

It does not establish:

- universal compressor generalization;
- zero-shot generalization;
- causal fault diagnosis;
- autonomous industrial control;
- plant-energy savings;
- carbon savings.
