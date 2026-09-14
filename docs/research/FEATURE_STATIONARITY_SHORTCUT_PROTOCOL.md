# Feature Stationarity / Shortcut Audit

## Status

This is a post-hoc reused-TEST diagnostic motivated by the score-transport
audit.

The prior audit found:

- global Robust-PCA raw-score transport was comparatively stable;
- both global Standard-PCA and regime Standard-PCA showed extreme
  CALIBRATION→TEST score inflation;
- the shift remained in background TEST outside registered incident windows;
- regime-PCA background residual mass was dominated by TP2;
- Standard-scaled drift rankings highlighted `tp2__min`, DV-pressure features,
  and pressure-switch features.

This audit diagnoses those features before any feature-removal or rescaling
candidate is allowed.

## Questions

1. Are the suspect features genuinely outside TRAIN support in raw units?
2. Is StandardScaler amplification caused by very small TRAIN standard
   deviations?
3. For analog sensors, are extreme minima isolated within-bin excursions or
   sustained changes also visible in mean/last values?
4. Did pressure-switch digital-state support/prevalence change?
5. Which exact individual engineered features dominate regime-PCA background
   residual score?
6. Are extreme feature values temporally isolated or clustered into long runs?

## Data

Allowed:

- TRAIN
- CALIBRATION
- TEST
- reported incidents

`background_test` is TEST outside the interval from 24 hours before each
reported incident start through that incident's end.

This does not imply background TEST is truly healthy.

## Frozen representation

Reconstruct the existing regime-conditioned Standard-PCA detector exactly:

- global StandardScaler fit on TRAIN;
- current/pressure q80 router fit on TRAIN;
- d=11 fixed regime PCA dimension;
- raw score = mean squared residual.

No model parameters are changed.

## Diagnostic A — StandardScaler denominator/support

For each focus feature report:

- TRAIN mean and StandardScaler scale;
- TRAIN raw min/q01/median/q99/max;
- CALIBRATION raw min/q01/median/q99/max;
- background TEST raw min/q01/median/q99/max;
- fraction of CALIBRATION below TRAIN min / above TRAIN max;
- fraction of background TEST below TRAIN min / above TRAIN max;
- fraction with |z| >= 5, 10, 50.

This distinguishes raw-support drift from merely large standardized values.

## Diagnostic B — analog minimum coherence

For `tp2` and `dv_pressure`, compare the five analog summaries:

- mean
- std
- min
- max
- last

Define an extreme-min row when:

`abs(z(sensor__min)) >= 10`

For those rows report fractions where:

- mean is also extreme at |z| >= 5;
- last is also extreme at |z| >= 5;
- both mean and last are non-extreme.

A large "min-only" fraction suggests brief within-bin excursions, spikes,
dropouts, or state-boundary effects rather than a sustained 5-minute shift.

No physical fault conclusion is made from this diagnostic alone.

## Diagnostic C — digital pressure-switch support

For TRAIN, CALIBRATION, and background TEST report for:

- `pressure_switch__last`
- `pressure_switch__active_ratio`
- `pressure_switch__transitions`

including:

- raw unique/support summary;
- nonzero fraction;
- for `last`, active fraction (>0.5);
- for `active_ratio`, any-active fraction (>0) and mostly-active fraction
  (>=0.5);
- for `transitions`, any-transition fraction (>0).

This tests whether the digital operating behavior itself changed.

## Diagnostic D — exact feature residual share

Use the existing exact per-feature regime-PCA squared residual contributions.

For:

- CALIBRATION all rows;
- background TEST all rows;
- background TEST rows above the CALIBRATION raw q99.5 threshold;

report contribution share by each of the 63 features.

Also report:

- top 20 individual features;
- total share from all `tp2__*` features;
- total share from all `dv_pressure__*` features;
- total share from all `pressure_switch__*` features;
- share from the explicit focus-feature set.

This is a detector-dependence diagnostic, not a causal claim.

## Diagnostic E — temporal clustering

For each focus feature and each configured |z| threshold, report in background
TEST:

- extreme-row count/fraction;
- first and last extreme timestamp;
- number of contiguous extreme runs;
- median/q90/q95/q99/max run rows;
- median/q90/q95/q99/max run duration.

This separates isolated feature excursions from sustained domain periods.

## Decision logic

- TP2 minimum dominates, with min-only excursions and normal mean/last:
  investigate feature-engineering robustness of extrema before any detector
  benchmark.
- TP2 minimum dominates and mean/last shift together:
  treat as a real sensor/state domain shift rather than a simple min-statistic
  artifact.
- pressure-switch support changes sharply with tiny TRAIN scale:
  investigate digital-feature stationarity / rare-state encoding.
- multiple related pressure features shift coherently:
  investigate broader sensor/state domain shift.
- no small feature subset explains residual mass:
  the Standard-PCA failure is broader than a shortcut feature.

No feature removal, clipping, or alternative scaling candidate is evaluated in
this audit.
