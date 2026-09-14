# AeroXAI Multi-Dataset Validation Protocol

Status: **research branch only**

This protocol must not alter the frozen `esic-2026-v3` release.

## Core research question

How stable are AeroXAI detector-evidence explanations across recording periods,
fault types, representations, scaling choices, assets, and sensing modalities?

Explanation concentration is measured with:

```text
C1     top-1 contribution share
C3     sum of the three largest contribution shares
H      Shannon entropy
H_norm H / log(K)
N_eff  exp(H)
HHI    sum(p_i^2)
```

These quantify **detector evidence**. They do not measure physical root-cause
count or causal fault complexity.

## MetroPT family

### MetroPT-3

Frozen baseline only. Do not retrain or retune it.

### MetroPT2

MetroPT2 is a related Porto Metro compressor benchmark, not an independent
cross-industry validation set.

The pre-registered MetroPT2 feature variants are:

```text
full                 68 features
flowmeter_excluded   63 features
strict_flow_blind    60 features
```

The frozen MetroPT2 alert protocol is:

```text
5-minute causal bins
train-only scaling / PCA fitting
PCA variance retained 0.95
calibration-only threshold q=0.995
EWMA alpha 0.20
persistence 3 / 4
10-minute reset gap
30-minute episode merge
24-hour early-warning window
2-hour late tolerance
```

Representation ablations and temporal/state analyses on MetroPT2 are
exploratory because the same test incidents have been inspected repeatedly.
They cannot promote a replacement primary detector.

## Independent external stress test: IITK Air Compressor

### Source and scope

Official IIT Kanpur Air Compressor Health State Dataset.

Eight health states are present:

```text
Healthy
Bearing
Flywheel
Leakage Inlet Valve (LIV)
Leakage Outlet Valve (LOV)
Non-Return Valve (NRV)
Piston ring
Rider belt
```

The downloaded archive contains 225 preprocessed recordings for every
condition. Local audit established that every `preprocess_ReadingN.dat`
contains exactly 50,000 finite scalar values.

This dataset is intentionally different from MetroPT:

```text
MetroPT     multivariate process telemetry
IITK        single-channel acoustic recordings
MetroPT     naturally timestamped incident periods
IITK        independent labelled recordings with seeded faults
```

Therefore IITK is **not** used for temporal precursor, lead-time, episode, or
fault-progression claims.

### Sampling-rate guardrail

The 50,000-value recording length is not treated as a verified sample rate.
Until the sampling rate of these exact preprocessed `.dat` files is established
from authoritative metadata, absolute frequency in Hz is not used.

Spectral locations are expressed only as:

```text
fraction of Nyquist
```

### Frozen recording split

Before any IITK detector score is viewed:

```text
Healthy Reading001-120   train
Healthy Reading121-180   calibration
Healthy Reading181-225   test
```

All 1,575 faulty recordings are **test only**.

Therefore:

```text
train        120 healthy
calibration   60 healthy
test          45 healthy + 1,575 faults = 1,620 recordings
```

No fault label may be used for fitting, scaling, PCA fitting, or threshold
calibration.

The healthy split is contiguous rather than random because neighboring
recording numbers may share unreported acquisition context.

### Frozen feature representation

Every 50,000-value waveform becomes one 31-feature vector.

Five explanation families are used:

```text
amplitude
distribution_shape
temporal_structure
spectral_shape
spectral_band_energy
```

The spectrum uses a Hann window after subtracting the recording mean.
Eight fixed equal-width bands divide the normalized Nyquist interval:

```text
[0.000, 0.125)
[0.125, 0.250)
[0.250, 0.375)
[0.375, 0.500)
[0.500, 0.625)
[0.625, 0.750)
[0.750, 0.875)
[0.875, 1.000]
```

Feature-family contribution is the **sum of detector contribution mass from
features assigned to that family**. Families contain unequal numbers of
features, so family size is part of the detector-evidence aggregation.
Cross-dataset interpretation therefore emphasizes normalized entropy and
within-dataset comparisons rather than raw N_eff alone.

### Frozen representation comparison

Evaluate exactly:

```text
RobustScaler   + PCA reconstruction
StandardScaler + PCA reconstruction
RobustScaler   + direct feature energy
StandardScaler + direct feature energy
```

PCA variance retained:

```text
0.95
```

Each representation receives its own threshold using **healthy calibration
recordings only**:

```text
q = 0.95
interpolation = higher
```

No EWMA, persistence, episode merge, or early-warning window is used.

### Frozen detection metrics

Global test composition is highly imbalanced:

```text
faults    1,575
healthy      45
fault prevalence = 0.972222...
```

Therefore global fault-positive average precision has a very high prevalence
baseline and must never be interpreted in isolation.

Primary reporting includes:

```text
ROC-AUC
average precision + explicit prevalence baseline
held-out healthy false-positive rate
overall fault recall
balanced accuracy
macro recall across the seven fault classes
per-fault recall
per-fault ROC-AUC versus the same 45 held-out healthy recordings
```

Per-fault ROC-AUC is especially important because it avoids allowing one easy
fault class to conceal another hard fault class.

### Frozen explanation metrics

For **every test recording**, not only detected faults, aggregate the 31 feature
contributions into the five frozen acoustic feature families and report:

```text
C1
C3
H_norm
N_eff
HHI
dominant-family frequency
```

Summaries are reported independently for Healthy and all seven fault classes.

Detected-fault-only explanation summaries may also be reported, but they are
secondary because conditioning on successful detection can bias explanation
statistics.

### Promotion and interpretation rules

Allowed:

> The representation generalized / failed to generalize as a recording-level
> novelty detector on an independent compressor/acoustic dataset.

Allowed:

> Explanation evidence was more concentrated or more distributed across
> acoustic feature families.

Not allowed:

> IITK validates MetroPT temporal precursor behavior.

Not allowed:

> A feature family is the physical root cause.

Not allowed:

> A seeded fault experiment proves production-site energy savings.

No IITK representation may replace the frozen V3 detector from this research
branch without a separate promotion decision and an additional independent
confirmatory evaluation.

## Post-benchmark uncertainty analysis

This section was frozen **after** the IITK external benchmark result was
observed. It is therefore exploratory follow-up analysis, not part of the
original confirmatory benchmark.

Its purpose is limited to quantifying finite-sample uncertainty and diagnosing
whether operating-point sensitivity is driven by the small healthy calibration
set.

No split, feature, representation, or test label may be changed. The operating
rule remains:

```text
healthy calibration only
q = 0.95
interpolation = higher
```

With 60 calibration recordings, NumPy's `higher` quantile at q=0.95 uses the
58th ordered observation (1-based), i.e. the third-highest calibration score.
This makes upper-tail sensitivity worth measuring explicitly.

The following analyses are frozen before their outputs are inspected:

```text
1. Wilson 95% reference intervals
   - held-out healthy false-positive rate
   - overall fault recall
   - each of seven per-fault recalls

2. Stratified non-parametric ROC-AUC bootstrap
   - 1,000 resamples
   - healthy and fault recordings resampled separately
   - overall ROC-AUC
   - per-fault ROC-AUC

3. Calibration-threshold sensitivity
   - leave-one-calibration-recording-out thresholds
   - 5,000 IID calibration bootstraps
   - 5,000 circular moving-block bootstraps, block length 5
   - 5,000 circular moving-block bootstraps, block length 10

4. For every calibration bootstrap threshold, apply that threshold to the
   unchanged IITK test scores and report the resulting:
   - held-out healthy FPR
   - overall fault recall
   - macro fault recall
   - balanced accuracy
   - per-fault recall
```

The moving-block analyses exist because adjacent recording numbers may share
unreported acquisition context. They are sensitivity analyses, not claims that
block lengths 5 or 10 are physically optimal.

The top six calibration anomaly scores and their healthy recording numbers are
reported to expose upper-tail leverage. They must not be removed or trimmed
after inspection.

Interpretation rules:

Allowed:

> The ranking representation is strong, while the finite calibration set makes
> the q=0.95 operating threshold sensitive / stable under resampling.

Allowed:

> Robust feature-energy exhibits good ranking but a conservative and
> calibration-sensitive operating point.

Only use the second statement if the frozen uncertainty results actually
support it.

Not allowed:

> A different threshold would be better because it improves IITK test recall.

Not allowed:

> Bootstrap confidence intervals are equivalent to an independent replication.

Not allowed:

> Perfect sampled ROC-AUC proves population ROC-AUC is exactly 1.
