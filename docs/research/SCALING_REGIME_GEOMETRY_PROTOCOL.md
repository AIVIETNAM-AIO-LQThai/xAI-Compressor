# AeroXAI Scaling and Regime Geometry Audit

Status: **research branch only**

This is a second-stage train-only diagnostic motivated by the preceding
healthy-manifold audit. It is not independent confirmation of that audit, but
its methods and thresholds are frozen before this second-stage result is run.

## Question

The previous exploratory geometry audit found:

- extremely low global Robust-PCA variance dimension;
- much larger TwoNN intrinsic dimension;
- local PCA reconstruction orders of magnitude below global PCA;
- very large tangent-plane rotation;
- a large reconstruction-error increase in high-current / high-pressure bins.

This audit asks which explanation best fits that pattern:

```text
scaling distortion
chronological distribution shift
distinct operating regimes
smooth within-regime curvature
or a mixture of these
```

## Data boundary

Only:

```text
data/processed/train.parquet
```

may be loaded.

The chronological split is frozen:

```text
first 80%  reference / fitting
last  20%  held-out within-training audit
```

No calibration data, test data, incident timestamps, or incident labels are
allowed.

## Frozen scaling comparison

Fit separately on the first 80%:

```text
RobustScaler
StandardScaler
```

For every feature and scaler, report:

```text
raw fit median
raw fit IQR
raw fit standard deviation
scaler scale parameter
fit scaled variance
audit scaled variance
maximum absolute scaled value
scaled Wasserstein distance from fit to audit
two-sample KS statistic
audit fraction outside fit [1%, 99%] range
```

For each scaler, fit one global PCA to the reference set and freeze the latent
dimension to the number of components required for 95% reference variance.

Report:

```text
95% component count
participation ratio
top absolute PC1 / PC2 / PC3 loadings
top held-out featurewise reconstruction-error contributors
```

The purpose is to test whether Robust scaling creates leverage from
small-IQR / heavy-tail features and whether the extreme geometry survives
Standard scaling.

## Frozen physical regime definition

The regime thresholds are determined only from the reference set:

```text
high current  = motor_current__mean >= reference 80th percentile
high pressure = reservoirs__mean   >= reference 80th percentile
```

This creates four interpretable regimes:

```text
low_current_low_pressure
high_current_only
high_pressure_only
high_current_high_pressure
```

The same frozen thresholds are then applied to the held-out training tail.

A regime is eligible for PCA comparison only if it contains at least:

```text
100 reference rows
25 held-out audit rows
```

These thresholds are frozen before results.

## Regime-conditioned PCA

Within each scaler separately:

1. fit global PCA on the whole reference set;
2. choose the global 95%-variance latent dimension;
3. for each eligible regime, fit a regime-specific PCA using **exactly the same
   latent dimension**;
4. compare held-out regime reconstruction MSE under:
   - the global PCA;
   - the corresponding regime PCA.

This isolates geometry rather than allowing regime-specific models to win
simply by using more dimensions.

Also report each regime's own 95%-variance dimension descriptively.

Evidence supporting regime structure would include:

```text
regime PCA << global PCA reconstruction error
```

for the same scaler and same latent dimension.

## Neighbor regime purity

For each scaler, find the 50 nearest reference neighbors of every held-out
training point.

Report:

```text
mean same-regime neighbor fraction
median same-regime neighbor fraction
fraction with >= 80% same-regime neighbors
```

High purity means local Euclidean neighborhoods align with the interpretable
current/pressure operating states.

## Subspace comparison

For each eligible regime:

- fit its full-reference PCA;
- split that regime's reference observations chronologically in half;
- when each half has at least 50 rows, fit a PCA to each half.

Using the frozen global latent dimension, report:

```text
within-regime early-vs-late maximum principal angle
between-regime maximum principal angles
```

Interpretation:

```text
small within-regime angle + large between-regime angle
    -> stable but distinct operating manifolds

large within-regime angle too
    -> smooth curvature, temporal drift, or unresolved sub-regimes remain
```

## Pre-stated interpretation gate

### Scaling artifact dominant

Supported if the extreme PCA leverage / residual regime effect largely
disappears under Standard scaling.

Next step:

```text
representation/scaling redesign
```

before nonlinear modeling.

### Regime structure dominant

Supported if both scalers show:

```text
high neighbor regime purity
regime-conditioned PCA improves held-out reconstruction
between-regime subspace angles exceed within-regime stability angles
```

Next step:

```text
regime-conditioned PCA / mixture-of-local-linear detector protocol
```

before a generic autoencoder.

### Within-regime nonlinearity remains

Supported if regime conditioning helps but substantial within-regime
subspace rotation / residual structure remains.

Next step:

```text
Kernel PCA / shallow autoencoder inside regimes
```

rather than one global nonlinear model.

## Guardrail

This checkpoint does not evaluate incidents, PR-AUC, alert timing, false
alerts, or any other anomaly-detection metric. The frozen V3 detector is not
changed.
