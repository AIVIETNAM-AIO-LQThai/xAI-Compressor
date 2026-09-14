# Regime-Transition EWMA Candidate Benchmark

## Status

This is an exploratory reused-test benchmark. The candidate was selected only
after TRAIN + CALIBRATION diagnostics showed strong cross-regime EWMA carryover.
The MetroPT-3 TEST set has already been inspected in earlier research and
therefore cannot provide independent confirmation.

## Question

Does one controlled temporal change:

> reset the EWMA state whenever the deterministic operating-regime label changes

retain the regime-conditioned Standard-PCA candidate's early detection and
ranking benefits while removing the calibration instability caused by
cross-regime state carryover?

## Exact comparison

1. Frozen Robust-PCA reference.
2. Regime-conditioned Standard-PCA with the original gap-reset-only EWMA.
3. Regime-conditioned Standard-PCA with transition-reset EWMA.

The regime-conditioned raw anomaly scores in (2) and (3) are identical.
Only temporal smoothing differs.

## Frozen detector

- 63 model features
- one global StandardScaler fitted on TRAIN
- `motor_current__mean` and `reservoirs__mean` q80 router from TRAIN
- four operating regimes
- global Standard-PCA 95% variance determines d
- every regime PCA uses the same d
- raw score = mean squared reconstruction residual

## Frozen alert protocol

- EWMA alpha = 0.20
- threshold quantile = 0.995, `higher`
- persistence = 3 of 4
- reset on time gap >10 minutes
- merge alert episodes within 30 minutes
- early warning window = 24 h
- late tolerance = 2 h
- 5-minute bins

### Candidate's only change

The EWMA recursion is reset to the current raw score when the regime label
changes.

Persistence history is intentionally **not** reset on regime change in this
experiment. This isolates the effect of EWMA state carryover.

Calibration and TEST smoothing are run independently.

## XAI

Exact squared-residual contributions are aggregated to the existing operator
signal groups.

For the candidate, every contribution column uses the same transition-reset
EWMA recursion as the scalar anomaly score. Therefore:

`sum(smoothed group contributions) == smoothed anomaly score`

up to floating-point tolerance.

Explanation concentration remains descriptive detector evidence, not physical
root-cause attribution.

## Reproduction gates

Before interpreting the candidate:

1. frozen Robust-PCA must reproduce its known metrics;
2. baseline regime Standard-PCA must reproduce its known metrics;
3. candidate calibration-only quantities discovered before this benchmark must
   reproduce:
   - q99.5 threshold = 0.12146337674041127
   - block-1 relative width = 0.5987219624925073
   - block-12 relative width = 0.6184667365721652

A reproduction failure stops the benchmark.

## Exploratory candidate gate

The candidate passes the preregistered exploratory gate only if all hold:

- timely incident recall >= 1.0
- pre-onset incident recall >= 1.0
- false alerts/day <= 1.6458665926285834
- PR-AUC >= 0.27670556668116664
- block-12 calibration relative width <= 0.67998060967883

This is deliberately stringent. Passing does not permit production promotion;
it only justifies independent confirmation on genuinely unseen evidence.

## Interpretation

Possible outcomes:

- retains 4/4 pre-onset, improves false alarms, preserves PR-AUC:
  transition-reset smoothing becomes the leading regime-PCA research candidate;
- calibration improves but TEST detection/ranking regresses:
  carryover was real but was also contributing useful temporal evidence;
- false alarms remain high:
  investigate persistence history or raw regime-score discontinuity next;
- PR-AUC/pre-onset collapses:
  do not promote the reset rule.

No post-result tuning is allowed.
