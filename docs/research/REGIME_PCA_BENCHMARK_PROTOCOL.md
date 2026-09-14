# AeroXAI Regime-Conditioned Standard-PCA Study

Status: **preregister before execution**

Target branch:

```text
research/regime-conditioned-pca
```

This study is motivated by the train-only geometry audits. The detector
benchmark itself must not be run until this protocol and its implementation are
committed and visible on GitHub.

## Why this study exists

The train-only geometry work produced two findings:

1. Robust scaling can create extreme leverage for small-IQR features and make
   the global PCA representation appear much lower-dimensional than the same
   data under Standard scaling.
2. Current/pressure operating regimes remain geometrically coherent under
   Standard scaling, and regime-conditioned PCA substantially improves
   held-out normal reconstruction.

The next controlled question is therefore:

> Does a piecewise-linear, regime-conditioned Standard-PCA representation
> improve the operational anomaly detector without sacrificing the exact
> reconstruction-error explanation mechanism?

This study does **not** yet test Kernel PCA, an autoencoder, polynomial feature
expansions, or any other generic nonlinear representation.

## Evidence status

The MetroPT-3 test period has already been inspected in previous work.

Therefore every result from this study must be labelled:

```text
EXPLORATORY_REUSED_TEST_BENCHMARK
```

Even a strong result is only a candidate for later independent confirmation.

## Frozen three-way comparison

Exactly three representations are evaluated:

```text
A. frozen Robust-PCA
B. global Standard-PCA
C. regime-conditioned Standard-PCA
```

A and B use the existing `PCADetector` implementation.

C uses the protocol below.

## Regime-conditioned representation

### Global scaler

Fit exactly one `StandardScaler` on the official training split using the same
63 model features as the frozen detector.

No regime-specific scaler is allowed.

This keeps all regime reconstruction errors in one comparable standardized
feature space.

### Router

The router uses two physically interpretable engineered features:

```text
motor_current__mean
reservoirs__mean
```

Thresholds are derived from the official training split only:

```text
high_current_threshold =
    training 80th percentile of motor_current__mean

high_pressure_threshold =
    training 80th percentile of reservoirs__mean
```

Four deterministic regimes are created:

```text
low_current_low_pressure
high_current_only
high_pressure_only
high_current_high_pressure
```

The thresholds are never re-estimated on calibration or test data.

### Latent dimension

Fit one global Standard-PCA on the training split.

Let:

```text
d_global
```

be the smallest component count that reaches at least 95% training explained
variance.

Every regime-specific PCA then uses exactly `d_global` components.

The regime models are therefore allowed different orientations but not a
different latent dimension.

This is a controlled piecewise-linear comparison rather than an
architecture-capacity search.

### Regime models

For every training observation:

1. Standard-scale using the one global scaler.
2. Route using the frozen current/pressure thresholds.
3. Fit one PCA to each regime using exactly `d_global` components.

All four regimes must be present in training and each must contain more than
`d_global` observations.

### Score

For a new observation:

1. apply the global StandardScaler;
2. assign the frozen operating regime from the raw engineered router features;
3. reconstruct with that regime's PCA;
4. compute

```text
score = mean_j((x_j - x_hat_j)^2)
```

over all model features.

Because every regime uses the same global scaler and the same feature count,
the primary study uses one pooled calibration threshold rather than tuning a
different alarm threshold per regime.

## Alert protocol

All three representations use the same frozen protocol:

```text
causal EWMA alpha        = 0.20
calibration quantile     = 0.995
quantile interpolation   = higher
persistence              = 3 hits in 4 bins
gap reset                = 10 minutes
episode merge            = 30 minutes
early warning window     = 24 hours
late tolerance           = 2 hours
bin width                = 5 minutes
```

Calibration and test EWMA sequences are computed independently.

No parameter may be changed after test results are visible.

## Frozen baseline reproduction guard

Before the regime candidate is interpreted, the script must reproduce the
known frozen Robust-PCA benchmark within `1e-12` for:

```text
timely incident recall
pre-onset incident recall
incident overlap recall
false alerts / 24h
time in alert
relevant episode precision
PR-AUC
```

Expected values are stored in the study configuration.

A reproduction failure invalidates the comparison.

## Detection metrics

Report for all three representations:

```text
timely incident recall
anytime incident recall
pre-onset incident recall
incident overlap recall
first timely alert per incident
lead / delay
episodes total
false episodes
false alerts per 24h
time in alert
relevant episode precision
PR-AUC
```

The four documented incidents remain the same MetroPT-3 high-stress air-leak
incidents used in the frozen benchmark.

## XAI protocol

For all three representations, the primary attribution remains exact
reconstruction residual energy:

```text
c_j = (x_j - x_hat_j)^2 / feature_count
```

Feature contributions are aggregated into the existing operator-level signal
groups.

At each first timely alert, report:

```text
dominant group
top-1 contribution share
top-3 contribution share
normalized entropy
effective group count
HHI
calibration-relative contribution percentile
```

For the regime detector, the explanation additionally records the active
operating regime.

The calibration percentile reference is pooled across the official calibration
split for this first controlled comparison.

This choice is frozen before seeing results. A later study may test
regime-specific rarity references, but not retroactively.

Explanation spread is descriptive. A more distributed explanation is not
automatically better, and a concentrated explanation is not automatically
worse.

No physical root-cause claim is permitted.

## Calibration-stability protocol

Alert-threshold stability is evaluated using only the calibration-smoothed
score sequence.

For each representation:

```text
2000 bootstrap replicates
seed = 20260914
95% interval
circular block lengths = 1 and 12
```

Block length 1 is the IID sensitivity reference.

Block length 12 corresponds to one hour of 5-minute bins and preserves short
local dependence more conservatively.

For each bootstrap mode report:

```text
median threshold
2.5th percentile
97.5th percentile
standard deviation
relative 95% width =
    (upper - lower) / abs(median)
```

This is a finite-calibration sensitivity diagnostic, not a confidence interval
for a population-optimal threshold.

## Pre-stated decision rule

No automatic production promotion is possible from this reused benchmark.

A **strict exploratory dominance** flag is true only if the regime-conditioned
candidate is simultaneously no worse than both global PCA references on:

```text
timely incident recall          higher is better
false alerts / 24h              lower is better
PR-AUC                          higher is better
block-12 relative threshold width lower is better
```

XAI concentration is reported separately and is not used as a monotonic
optimization target.

If strict dominance fails, the candidate may still be scientifically
interesting, but no detector replacement claim is allowed.

If strict dominance succeeds, the next step is independent confirmation on a
separate telemetry benchmark without retuning.

## What this study cannot establish

Do not claim:

- regime routing identifies a physical fault mode;
- one regime PCA is a causal model of compressor physics;
- improved reconstruction automatically means better anomaly detection;
- improved test metrics are independent confirmation;
- a test-set win replaces the frozen V3 detector;
- explanation concentration equals physical fault complexity.

## Next-step gate

After results:

```text
candidate clearly fails
    -> stop regime detector escalation

candidate trades metrics
    -> analyze why without retuning

candidate strictly dominates
    -> freeze candidate and move to independent telemetry confirmation

within-regime evidence still shows systematic geometry problems
    -> only then preregister nonlinear-within-regime models
```
