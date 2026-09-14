# Transition-Reset Candidate Failure Audit

## Status

This is a **post-hoc reused-test failure diagnostic**.

The transition-reset EWMA candidate has already been evaluated on the reused
MetroPT-3 TEST set and failed its preregistered gate. Therefore this audit is
not independent validation and cannot rescue the candidate.

The audit freezes only the diagnostic procedure used to explain the already
observed failure.

## Observed failure to explain

The candidate improved calibration stability but:

- increased total episodes and false episodes;
- increased false alerts/day;
- reduced PR-AUC relative to baseline regime PCA;
- reduced pre-onset recall from 4/4 to 3/4;
- reduced total time in alert despite creating many more episodes.

That pattern is compatible with alert fragmentation, persistence windows that
cross regime boundaries, calibration-to-TEST score shift, or a mixture.

## Data allowed

This audit may load:

- TRAIN
- CALIBRATION
- TEST
- reported incidents

TEST and incident use is allowed only because their results have already been
observed. Every conclusion remains exploratory/post-hoc.

## Frozen candidate

Reuse the already implemented:

- regime-conditioned Standard-PCA raw detector;
- q80 current/pressure router;
- d=11 regime PCAs;
- transition-reset EWMA;
- pooled q=.995 calibration threshold;
- 3-of-4 persistence;
- 10-minute reset gap;
- 30-minute episode merge.

No alternative alert rule is evaluated.

## Reproduction gate

Before diagnostics:

1. reproduce the baseline regime-PCA benchmark metrics;
2. reproduce the transition-reset candidate threshold and benchmark metrics.

A mismatch stops the audit.

## Diagnostic A — calibration-to-TEST score shift

For transition-reset EWMA, report pooled and per-regime CALIBRATION vs TEST:

- count
- median
- q90
- q95
- q99
- q99.5
- maximum
- exceedance fraction above the candidate calibration threshold

Report TEST/CALIBRATION q99 and q99.5 ratios where defined.

This tests whether a stable calibration quantile is nevertheless poorly
transported to TEST.

## Diagnostic B — regime occupancy and transition frequency

Compare CALIBRATION vs TEST:

- regime fractions;
- transition count;
- transition fraction;
- transitions per observed day.

This tests whether the TEST temporal regime process differs materially from
CALIBRATION.

## Diagnostic C — exact persistence-window composition

Reconstruct the existing 3-of-4 persistence history without changing it.

For every TEST row record:

- threshold hit;
- alert state;
- number of hits in active persistence history;
- active history length;
- number of distinct regimes in the persistence window;
- number of regime changes within the persistence window;
- whether the persistence window spans more than one regime.

Time gaps >10 minutes clear history exactly as the existing implementation does.

No persistence-reset alternative is simulated.

## Diagnostic D — alert-start mechanism

For every alert episode, inspect the persistence window at the episode start.

Report:

- whether the start window spans multiple regimes;
- number of regimes in the start window;
- number of regime transitions in the start window;
- whether episode start is exactly at / within 1 / within 2 rows of a regime
  transition;
- start regime;
- start EWMA score / threshold ratio;
- alert-bin count and episode duration.

## Diagnostic E — false vs relevant episode comparison

Classify relevant episodes using the exact existing evaluator definition:

An episode is relevant if it overlaps the interval from 24 hours before any
incident start through that incident's end.

All other episodes are false.

Compare false vs relevant episodes on the alert-start diagnostics above.

## Diagnostic F — fragmentation

For baseline regime PCA and transition-reset candidate report:

- total episodes;
- false episodes;
- median / q90 / q95 / q99 alert bins per episode;
- episode duration distribution;
- inter-episode gap distribution;
- fraction of one-bin / <=2-bin / <=3-bin episodes.

This quantifies whether the candidate's higher false-episode count mainly
reflects fragmentation into many short episodes.

## Interpretation

Possible next steps:

- false episode starts overwhelmingly use mixed-regime persistence windows:
  preregister a persistence-reset-at-regime-change candidate;
- TEST exceedance rates are much larger than CALIBRATION even within regimes:
  investigate calibration transport / conditional calibration rather than
  persistence;
- false episodes are mostly transition-adjacent but persistence windows remain
  single-regime:
  investigate raw router discontinuity / transition-aware gating;
- no dominant mechanism:
  accept that transition-reset EWMA fixed calibration uncertainty but is
  operationally inferior on reused TEST.

No intervention is executed in this audit.
