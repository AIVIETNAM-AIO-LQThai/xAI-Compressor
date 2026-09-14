# Coverage-Conditioned Stationarity Audit

## Status

This is a post-hoc reused-TEST diagnostic.

The preceding operating-cycle support audit found that TRAIN-support violations
are strongly associated with the regime-PCA background score, but those same
rows also have materially lower `coverage_ratio` and `sample_count`.

Therefore the next question is deliberately upstream of any router redesign:

> Do the TP2 / DV-pressure / pressure-switch support violations persist on
> complete, well-covered aggregation windows?

No detector intervention is evaluated here.

## Frozen detector and support definition

Reconstruct the existing regime-conditioned Standard-PCA detector exactly:

- 63 model features;
- global StandardScaler fit on TRAIN;
- current/reservoir q80 router fit on TRAIN;
- four router regimes;
- d=11 regime PCAs;
- raw score = mean squared residual.

Recreate the exact TRAIN-min/max support flags from the preceding audit.

The CALIBRATION raw q99.5 threshold remains descriptive only.

## TEST partition

`relevant_test` is from 24 hours before each registered incident through the
incident end.

`background_test` is every other TEST row.

Background does not imply confirmed health.

## Frozen quality strata

Coverage strata:

- `full`: `coverage_ratio >= 1.0`
- `partial_high`: `0.90 <= coverage_ratio < 1.0`
- `partial_low`: `coverage_ratio < 0.90`

Sample-count strata:

- `complete`: `sample_count >= 30`
- `mild_underfill`: `27 <= sample_count <= 29`
- `severe_underfill`: `sample_count <= 26`

These cutoffs are frozen before this audit is run.

## Diagnostic A — novelty by data quality

For every atomic and composite support flag report prevalence within each
coverage stratum and each sample-count stratum.

Also report:

- fraction of novel rows that are full coverage;
- fraction of supported rows that are full coverage;
- fraction of novel rows with complete sample count;
- fraction of supported rows with complete sample count.

## Diagnostic B — score by quality × novelty

In background TEST, create the four primary cells:

1. full coverage + supported;
2. full coverage + any-cycle-novel;
3. partial coverage + supported;
4. partial coverage + any-cycle-novel.

For each cell report:

- rows/fraction;
- median/q90/q95/q99/q99.5/max raw regime-PCA score;
- fraction above the frozen CALIBRATION raw q99.5 threshold;
- share of total background raw-score mass.

Repeat inside each of the four existing router regimes.

This tells us whether quality alone, novelty alone, or their interaction drives
the score explosion.

## Diagnostic C — calibration transport under matched quality

Compare raw regime-PCA score distributions for:

- CALIBRATION full coverage;
- background TEST full coverage;
- CALIBRATION complete sample count;
- background TEST complete sample count.

Report q99/q99.5 ratios and threshold exceedance fractions.

If transport remains poor after matching quality, the support shift is not
explained by underfilled bins alone.

## Diagnostic D — focus-feature raw support under full coverage

For the six frozen focus features, compare TRAIN, CALIBRATION, background TEST
full-coverage, and background TEST partial-coverage raw distributions.

Report:

- min/q01/median/q99/max;
- fraction below TRAIN min;
- fraction above TRAIN max.

This directly tests whether TP2/DV-pressure support violations remain in
complete windows.

## Diagnostic E — overlap strength

For each support flag calculate in background TEST:

- P(flag | partial coverage)
- P(flag | full coverage)
- risk ratio
- odds ratio with 0.5 Haldane-Anscombe correction when necessary.

Repeat for sample-count underfill vs complete.

These are descriptive associations, not causal estimates.

## Diagnostic F — joint patterns at full coverage

Recompute the six atomic joint support patterns using only background TEST
full-coverage rows.

Report the top 20 patterns overall and among high-score rows.

This shows whether the previously dominant cycle patterns survive once
underfilled windows are excluded descriptively.

## Decision logic

- support violations and score transport largely disappear at full coverage:
  next study should target aggregation / missing-sample robustness, not a richer
  router;
- support violations remain common at full coverage and still dominate scores:
  evidence supports a genuine missing operating-state representation;
- analog novelty survives but digital novelty disappears:
  pursue analog operating-state subdivision;
- digital novelty survives at full coverage:
  investigate pressure-switch state representation;
- full-coverage TEST still shifts strongly even without support violations:
  the current flag set is incomplete and broader stationarity analysis is
  needed.

No rows are removed and no candidate detector is benchmarked in this audit.
