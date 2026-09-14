# Regime-Transition EWMA Carryover Audit

## Status

This is a post-hoc, calibration-only diagnostic prompted by the previously
frozen regime calibration/routing audit.

The preceding audit showed:

- train/calibration regime occupancy is nearly unchanged;
- bootstrap threshold variation has weak correlation with regime fractions;
- router transitions are extremely frequent;
- raw score jumps are materially larger at regime transitions;
- pooled EWMA is highly autocorrelated;
- conditional EWMA upper tails can be radically different from the raw-score
  upper tails.

This study asks one narrow question:

> Does carrying a single EWMA state across operating-regime changes materially
> inflate or redistribute calibration tails and threshold instability?

## Data boundary

Allowed:

- `data/processed/train.parquet`
- `data/processed/calibration.parquet`

Forbidden:

- TEST
- incident labels
- incident timestamps
- any test-derived performance metric

## Frozen detector

Reuse the existing regime-conditioned Standard-PCA implementation unchanged.

Frozen detector identity:

- one global StandardScaler fitted on TRAIN;
- current/pressure q80 router fitted on TRAIN;
- four regime-specific PCAs;
- global Standard-PCA 95% variance fixes d=11 for all regimes;
- raw score is mean squared reconstruction residual.

## Smoothing comparison

Exactly two smoothing sequences are compared.

### A. Baseline pooled EWMA

Existing production-style research behavior:

- alpha = 0.20
- reset only when timestamp gap >10 minutes
- regime changes do not reset EWMA state

### B. Diagnostic transition-reset EWMA

Same alpha and same time-gap reset, but also reset the EWMA state when the
operating-regime label changes.

This is not a promoted candidate. It is a mechanism test.

No other smoothing variant may be introduced after seeing the result.

## Primary diagnostics

1. Reproduce the baseline pooled q99.5 threshold exactly.
2. Count regime transitions and reproduce the preceding diagnostic count.
3. At every regime transition, record:
   - previous regime
   - current regime
   - raw score
   - baseline EWMA
   - transition-reset EWMA
   - baseline minus reset EWMA
   - absolute difference
4. Summarize carryover by transition pair.
5. Compare by-regime raw, baseline-EWMA and reset-EWMA q99/q99.5 values.
6. Compare pooled q99/q99.5 tail composition under both smoothers.
7. Compare lag autocorrelation under both smoothers.
8. Bootstrap the already-smoothed calibration sequences using the same paired
   circular-block resampling definition and report threshold relative width for
   block lengths 1 and 12.

## Interpretation

Evidence for cross-regime EWMA carryover is strengthened if transition-reset
EWMA materially:

- reduces extreme discrepancy between raw and smoothed conditional tails;
- reduces transition-row baseline-vs-reset differences;
- reduces short-lag autocorrelation;
- narrows block-bootstrap threshold width.

Even a strong result does not justify immediate production use. A later
preregistered candidate benchmark would be required before applying the
transition-reset smoother to TEST.

## Outputs

Tracked:

- `docs/research/regime_ewma_carryover_audit.json`

Local only:

- `data/processed/regime_ewma_carryover_audit/transition_carryover.parquet`
- `data/processed/regime_ewma_carryover_audit/bootstrap_thresholds.parquet`
