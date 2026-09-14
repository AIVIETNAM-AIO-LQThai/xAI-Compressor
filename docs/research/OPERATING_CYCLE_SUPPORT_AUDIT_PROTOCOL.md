# Operating-Cycle Support Audit

## Status

This is a post-hoc reused-TEST diagnostic.

The preceding feature-stationarity audit showed that the largest Standard-PCA
transport failures are not adequately explained as isolated scaler artifacts:

- TP2 extreme-min rows usually also have an extreme TP2 mean;
- DV-pressure extreme-min rows almost always have an extreme DV-pressure mean;
- pressure-switch transition behavior changes materially in background TEST;
- the exact regime-PCA residual is dominated by TP2 minimum behavior.

The next question is whether these features describe a **new operating-cycle
state that the existing current/reservoir four-regime router does not capture**.

No new router is tested here.

## Frozen detector

Reconstruct the existing regime-conditioned Standard-PCA detector exactly:

- 63 model features;
- StandardScaler fitted on TRAIN;
- current/reservoir q80 router fitted on TRAIN;
- four regimes;
- d=11 fixed PCA dimension;
- raw score = mean squared reconstruction residual.

## TEST partition

`relevant_test` is from 24 hours before each registered incident through the
incident end.

`background_test` is every other TEST row.

Background does not imply confirmed health.

## TRAIN-derived support flags

All novelty thresholds are frozen from TRAIN support only.

For each background/CALIBRATION row, define:

- `tp2_min_above_train_max`
- `tp2_mean_above_train_max`
- `dv_pressure_min_above_train_max`
- `dv_pressure_mean_above_train_max`
- `pressure_switch_transitions_above_train_max`
- `pressure_switch_active_ratio_below_train_min`

Composite flags:

- `analog_cycle_novel` = any analog support violation;
- `digital_cycle_novel` = any pressure-switch support violation;
- `any_cycle_novel` = analog OR digital.

These are support diagnostics, not anomaly labels.

## Diagnostic A — support prevalence

For CALIBRATION, background TEST and relevant TEST report:

- count/fraction for every atomic flag;
- count/fraction for every composite flag;
- overlap between analog and digital novelty.

If CALIBRATION is near-zero but background TEST is substantial, the operating
support changed after the calibration period.

## Diagnostic B — existing-router coverage

For every atomic/composite flag in background TEST report:

- regime distribution among flagged rows;
- flag prevalence within each of the four existing regimes.

If the novel cycle state appears materially inside several existing regimes,
current/reservoir routing is not sufficient to isolate it.

## Diagnostic C — score association

Use the frozen regime-PCA raw CALIBRATION q99.5 threshold.

For every flag in background TEST report:

- score median/q90/q95/q99/q99.5/max among flagged rows;
- threshold exceedance fraction among flagged rows;
- fraction of all high-score background rows carrying the flag;
- share of total background raw score mass carried by flagged rows.

This quantifies how strongly each unsupported state drives detector failure.

## Diagnostic D — joint support patterns

Encode the six atomic support flags into exact joint patterns.

Report the top 20 patterns in:

- all background TEST;
- high-score background TEST.

For each pattern report row count/fraction, high-score fraction, mean score,
and regime distribution.

This distinguishes one dominant missing state from several unrelated drifts.

## Diagnostic E — temporal structure

For every atomic/composite flag in background TEST report contiguous run
statistics:

- run count;
- median/q90/q95/q99/max rows;
- median/q90/q95/q99/max duration;
- first/last timestamp.

Long runs support a persistent operating-state interpretation; mostly
single-bin runs support transient feature excursions.

## Diagnostic F — data-quality check

Compare `sample_count` and `coverage_ratio` across TRAIN, CALIBRATION and
background TEST.

Also compare these quality variables between background rows with and without
`any_cycle_novel`.

This checks whether the apparent operating-state shift could instead be
explained by a sampling/coverage change.

## Decision logic

- novel-cycle rows dominate high scores and appear across multiple existing
  regimes, while quality is stable:
  next study should examine a richer operating-state router / cycle-state
  representation.
- novelty is concentrated in one existing regime:
  investigate within-regime state subdivision there.
- novelty is mostly one-bin and tied to low coverage/sample count:
  investigate preprocessing/aggregation robustness first.
- novelty is persistent but weakly associated with high score:
  the Standard-PCA failure has another upstream cause.
- several disjoint patterns dominate:
  do not force a single new regime; investigate multi-state representation.

No intervention is executed in this audit.
