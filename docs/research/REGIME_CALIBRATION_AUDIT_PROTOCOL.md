# Regime Calibration and Routing Instability Audit

## Status

This study is a **post-hoc diagnostic** prompted by the already-observed
Regime-Conditioned Standard-PCA benchmark.

The benchmark result is already known. Therefore this study must **not** be
described as part of the original regime-PCA preregistration.

What is frozen before running this audit is the **diagnostic procedure below**.

## Research question

The regime-conditioned Standard-PCA candidate improved ranking and pre-onset
incident coverage relative to global Standard-PCA, but its pooled calibration
threshold was much less stable than the frozen Robust-PCA detector.

This audit asks which mechanisms are compatible with that calibration failure:

1. incompatible score scales across operating regimes;
2. rare-regime or rare-tail domination of the pooled threshold;
3. hard-router score discontinuities near regime changes;
4. train-to-calibration regime occupancy shift;
5. temporal clustering / autocorrelation;
6. instability that remains even after conditioning on operating regime.

The audit is diagnostic only. It does not attempt to rescue the candidate.

## Data boundary

Allowed:

- `data/processed/train.parquet`
- `data/processed/calibration.parquet`

Forbidden:

- `data/processed/test.parquet`
- reported incident labels
- reported incident timestamps
- any test-derived model-selection logic

The script must print that TEST and incident labels are not loaded.

## Frozen candidate identity

The detector must be reconstructed by reusing the existing implementation:

- `fit_regime_pca_detector`
- `score_regime_pca_detector`
- `assign_regimes`

Frozen representation:

- 63 model features
- one global StandardScaler fitted on training data
- router variables:
  - `motor_current__mean`
  - `reservoirs__mean`
- router high-state quantile: 0.80, derived from training only
- four operating regimes
- global Standard-PCA 95% variance determines one fixed component count
- every regime PCA uses that same component count
- reconstruction score is mean squared residual

Frozen alert-calibration representation:

- causal EWMA alpha = 0.20
- reset gap = 10 minutes
- pooled calibration quantile = 0.995
- quantile interpolation/method = `higher`

No existing detector or benchmark source file may be edited for this audit.

## Reproduction gate

Before diagnostic interpretation, the script must reproduce the known
regime-PCA candidate identity:

- train rows = 5,271
- calibration rows = 1,796
- features = 63
- PCA components = 11
- global explained variance = 0.9576052415845236
- current threshold = 3.3312096774193547
- pressure threshold = 9.452193548387097
- pooled calibration threshold = 0.25200852435862636
- train regime counts:
  - low current / low pressure = 3,881
  - high current only = 335
  - high pressure only = 335
  - high current / high pressure = 720

The existing calibration-stability procedure must also reproduce the known
relative widths for circular blocks 1 and 12 within the configured tolerance.

Failure of the reproduction gate stops the audit. The frozen values may not be
changed to make the audit pass.

## Diagnostic 1: regime occupancy

For TRAIN and CALIBRATION, report each regime's:

- count
- fraction
- calibration-minus-train fraction change

This tests whether the mixture of operating states changed between the two
splits.

## Diagnostic 2: conditional score distributions

For pooled calibration and each regime, summarize both raw reconstruction score
and the actual pooled causal-EWMA score:

- count
- mean
- standard deviation
- median
- IQR
- q90
- q95
- q99
- q99.5
- maximum

Large differences in conditional upper-tail scale are evidence that one pooled
quantile is mixing heterogeneous score distributions.

## Diagnostic 3: pooled upper-tail composition

For the pooled q99 and q99.5 EWMA tails, report:

- pooled cutoff
- number of tail rows
- regime composition of tail rows
- ordinary calibration regime composition
- enrichment ratio

For regime `r`:

`enrichment_r = tail_fraction_r / calibration_fraction_r`

This quantifies whether one regime disproportionately controls the pooled tail.

## Diagnostic 4: router margins and transition discontinuity

For every calibration row compute raw router margins:

- current value - current threshold
- pressure value - pressure threshold

Normalize each margin by the corresponding raw training standard deviation.

Define router-boundary distance as the smaller absolute normalized margin.

A transition occurs when the regime differs from the previous observation and
the observations belong to the same <=10-minute contiguous segment.

For transition and non-transition rows compare absolute raw-score and EWMA-score
jumps using median, q90, q95, and q99.

Save windows of +/-2 rows around each transition without crossing temporal
segments.

## Diagnostic 5: tail proximity to router transitions

For pooled q99 and q99.5 rows report the fractions:

- exactly at a transition
- within +/-1 row of a transition
- within +/-2 rows of a transition
- not within +/-2 rows

Compare those fractions with ordinary calibration prevalence and report
enrichment.

## Diagnostic 6: regime runs and temporal persistence

For each regime report contiguous run:

- number of runs
- mean length
- median length
- q90 length
- maximum length

Also report total transitions and transitions per observed day.

Calculate gap-aware score autocorrelation at lags 1, 2, 3, 6, and 12 for:

- pooled EWMA score
- each regime where sufficient paired observations exist

Pairs may not cross a >10-minute reset gap.

## Diagnostic 7: bootstrap mixture sensitivity

Reproduce the frozen circular-block bootstrap procedure with paired
(score, regime) observations.

For every bootstrap replicate save:

- threshold
- fraction of each regime in the resample

For each block length, calculate Spearman correlation between regime fraction
and threshold.

This tests whether variation in bootstrap regime composition tracks variation
in the pooled threshold.

## Diagnostic 8: conditional thresholds

Report each regime's q99.5 value from the **already-computed pooled EWMA series**.

These are explicitly:

- diagnostic conditional thresholds;
- not alert thresholds;
- not applied to TEST;
- not a new detector.

## Outputs

Tracked research output:

- `docs/research/regime_calibration_audit.json`

Local, untracked detailed tables:

- `data/processed/regime_calibration_audit/calibration_points.parquet`
- `data/processed/regime_calibration_audit/transition_windows.parquet`
- `data/processed/regime_calibration_audit/bootstrap_thresholds.parquet`

The Parquet tables are analysis artifacts and must not be committed at this
checkpoint.

## Interpretation

The audit must report evidence, not automatically promote an intervention.

Possible follow-up studies are separated:

- stable conditional distributions + unstable pooled threshold:
  preregister conditional-calibration study;
- tail concentrated around regime transitions:
  preregister soft/gated-routing study;
- rare regime dominates tail:
  preregister calibration-mixture / sampling study;
- strong instability remains within regimes:
  revisit within-regime geometry and only then consider nonlinear models.

No hyperparameter may be changed after this diagnostic result is observed.
