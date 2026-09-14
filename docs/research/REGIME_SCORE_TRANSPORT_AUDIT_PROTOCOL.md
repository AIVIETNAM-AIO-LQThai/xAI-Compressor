# Regime-PCA Score Transport Audit

## Status

This is a post-hoc reused-TEST diagnostic. It follows the failed
transition-reset-EWMA benchmark and the transition-alert failure audit.

The previous audit established two facts:

1. alert fragmentation is strongly associated with regime boundaries;
2. a much larger issue exists upstream: the transition-reset regime-PCA score
   distribution shifts dramatically from CALIBRATION to TEST, including within
   the same router regimes.

This audit isolates the upstream representation problem before any further
alert-rule intervention.

## Main question

Why does the anomaly-score distribution fail to transport from CALIBRATION to
TEST?

Three frozen representations are compared:

1. global Robust-PCA;
2. global Standard-PCA;
3. regime-conditioned Standard-PCA.

All models are fit on TRAIN only.

## Data use

Allowed:

- TRAIN
- CALIBRATION
- TEST
- reported incidents

This is explicitly reused-TEST/post-hoc evidence.

## Raw-score transport

For each representation:

- fit on TRAIN;
- compute raw reconstruction score on CALIBRATION and TEST;
- derive a descriptive raw q99.5 threshold from CALIBRATION;
- report CALIBRATION and TEST median/q90/q95/q99/q99.5/max;
- report TEST fraction above the CALIBRATION q99.5 threshold;
- report TEST/CALIBRATION q99 and q99.5 ratios.

No EWMA, persistence or episode logic is used in the primary transport
comparison.

For regime Standard-PCA, repeat transport diagnostics inside each of the four
router regimes.

## Known-event vs background TEST

Define `relevant_test` as rows from 24 hours before each reported incident
start through that incident's end.

Define `background_test` as every TEST row outside those windows.

Report score transport separately for:

- all TEST;
- relevant TEST;
- background TEST.

If the huge shift persists in background TEST, it cannot be explained solely
by the four reported incidents.

This does not mean background TEST is guaranteed healthy; it means only that it
is outside the registered relevant windows.

## Feature drift under the frozen StandardScaler

Using the StandardScaler fit on TRAIN for regime PCA, transform CALIBRATION and
background TEST.

For each of the 63 model features report:

- CALIBRATION median absolute z;
- CALIBRATION q99 absolute z;
- background TEST median absolute z;
- background TEST q99 absolute z;
- TEST/CALIBRATION q99-|z| ratio;
- fraction with |z| >= 5;
- fraction with |z| >= 10;
- fraction with |z| >= 50.

Rank the top 20 features by background TEST q99 absolute z and by q99 ratio.

This tests whether Standard scaling is extrapolating far outside the TRAIN /
CALIBRATION support.

## Contribution-group transport

Use exact regime-PCA squared-residual contributions and existing operator
groups.

For CALIBRATION, background TEST, and relevant TEST report aggregate
contribution shares:

- over all rows;
- over rows above the regime-PCA CALIBRATION raw q99.5 threshold.

This identifies which signal groups dominate transported high scores.

These are detector-evidence contributions, not physical root-cause claims.

## Background high-score runs

Within background TEST, identify contiguous runs whose regime-PCA raw score is
at or above the CALIBRATION raw q99.5 threshold.

Report:

- run count;
- median/q90/q95/q99/max duration;
- median/q90/q95/q99/max rows;
- start regime distribution.

This distinguishes isolated outliers from long unlabelled/domain-shift
intervals.

## Decision logic

- Robust stable, global Standard + regime Standard unstable:
  Standard-scaling / feature-stationarity problem.
- global Standard stable, regime Standard unstable:
  regime-local PCA extrapolation / router representation problem.
- all three unstable:
  broad TEST-domain or unlabelled-state shift.
- background TEST stable but relevant TEST unstable:
  shift is mostly explained by known incident windows.
- background TEST remains strongly shifted:
  do not fix the problem by alert-rule tuning; investigate representation/data
  stationarity first.
- one or a few features/groups dominate:
  follow with a preregistered feature-stationarity/shortcut audit.

No intervention is executed in this audit.
