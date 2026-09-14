# AeroXAI Healthy-Manifold Geometry Audit

Status: **research branch only**

This protocol is frozen before the geometry results are inspected.

## Research question

Does healthy MetroPT-3 compressor operation occupy an approximately linear
low-dimensional manifold, or does global PCA flatten meaningful nonlinear or
multi-regime geometry?

The frozen anomaly detector remains unchanged.

## Data boundary

This checkpoint may read only:

```text
data/processed/train.parquet
```

It must not load calibration or test data and must not use incident timestamps
or labels.

The MetroPT training set is split chronologically:

```text
first 80%  geometry reference / fit
last  20%  held-out within-training geometry audit
```

The RobustScaler is fit only on the first 80%.

## Frozen diagnostics

1. Full global-PCA variance spectrum and components needed for 80%, 90%, 95%,
   and 99% variance.
2. Participation-ratio intrinsic dimension.
3. Two-nearest-neighbor intrinsic dimension.
4. Global-vs-local PCA reconstruction at k = 25, 50, 100, 200.
5. Local/global and consecutive-local tangent-space principal-angle summaries.
6. Global-PCA residual dependence on PC1/PC2/PC3 and pre-registered operating
   features.
7. Held-out relationship-shape comparison using linear, degree-2 polynomial,
   and cubic-spline regression.

The local-PCA latent dimension is frozen to the number of global components
required for 95% reference-set variance.

### Participation ratio

```text
D_PR = (sum(lambda_i))^2 / sum(lambda_i^2)
```

### Two-nearest-neighbor estimate

```text
r1 = first non-self neighbor distance
r2 = second non-self neighbor distance
mu = r2 / r1
D_2NN = 1 / mean(log(mu))
```

Degenerate distance cases are excluded and counted.

### Local PCA interpretation

For each held-out within-training point, fit PCA to its k nearest reference
points and compare reconstruction MSE with the frozen global PCA.

Systematic local improvement supports the hypothesis that one global plane is
insufficient, but does not prove that a nonlinear anomaly detector will perform
better.

### Relationship-shape diagnostics

The following directional pairs are frozen:

```text
tp2 mean             -> tp3 mean
tp3 mean             -> reservoirs mean
motor current mean   -> reservoirs mean
motor current mean   -> tp3 mean
dv pressure mean     -> reservoirs mean
oil temperature mean -> motor current mean
```

Evaluate on the held-out 20%:

```text
linear regression
degree-2 polynomial regression
cubic spline with 6 knots
```

These are geometry diagnostics, not causal models.

## Decision gate

No Kernel PCA, autoencoder, local-PCA anomaly detector, or other nonlinear
detector is trained here.

Possible outcomes:

```text
A. Approximately flat
   -> stop nonlinear-model escalation.

B. Low-dimensional but curved
   -> preregister Kernel PCA / local PCA / shallow autoencoder comparison.

C. Strong multi-regime geometry
   -> prioritize interpretable regime-conditioned or mixture-of-local-linear
      models before generic high-capacity nonlinear models.
```

The existing product PCA remains frozen regardless of this result.
