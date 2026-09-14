# Evidence-Aware Compute Routing

Status: **PREREGISTRATION — no evidence-aware router result executed**

Branch:

`research/evidence-aware-compute-routing`

Frozen parent failure commit:

`7561a8acdd1e9e03fd6212272bae95908bb3dd45`

## 1. Why a new study is required

The preceding support-aware study failed its preregistered development
criterion with 0/8 eligible policies.

That failure was structural rather than merely a poor support-boundary
choice.

The previous controller could choose only q90, q95 or q99 PCA thresholds.
Because q90 was the least restrictive admissible threshold,

\[
R_{\mathrm{adaptive}} \subseteq R_{q90}.
\]

Yet fixed q90 covered only 20.690% of MetroPT-3 CAL high-TCN-evidence bins
and 10.432% of MetroPT2 CAL high-TCN-evidence bins.

Therefore no member of that router family could reach the preregistered
80% evidence target.

This study starts a new router family instead of modifying the failed one.

## 2. Research question

> Can a cheap calibration-trained escalation model predict when temporal
> inference is valuable, achieving at least 80% TCN evidence coverage while
> minimizing TCN invocation under cross-dataset support shift?

The router may invoke the TCN even when PCA score is below q90.

## 3. Development evidence

Development is restricted to:

- MetroPT-3 TRAIN;
- MetroPT-3 CALIBRATION;
- MetroPT2 TRAIN;
- MetroPT2 CALIBRATION.

MetroPT-3 TEST and MetroPT2 TEST are explicitly forbidden for:

- router feature design;
- router fitting;
- model selection;
- probability-threshold selection;
- eligibility decisions.

Incident labels are also forbidden during router development.

## 4. Frozen TCN identities

MetroPT-3 TCN:

`3a2a98fd65ae91014aba89b04b953b984d6763003351ec03724d682632bc8f60`

MetroPT2 TCN:

`8709a5ce5e08225b2573e95960086c1e7641c31f752f656b9e50c0b23a7e2bf0`

The TCN remains second-stage research evidence and does not modify the
operational Robust-PCA alert.

## 5. Chronological router development split

For each dataset, first construct the valid CAL target rows produced by the
existing exact causal TCN sequence rule.

Let the number of valid CAL TCN targets be \(N\).

The split point is

\[
n_{\mathrm{fit}} = \lfloor 0.60N \rfloor.
\]

The first \(n_{\mathrm{fit}}\) chronological target rows are the
**router-fit** subset.

The remaining rows are the **router-validation** subset.

There is no shuffle.

Causal rolling features on the first validation rows may use earlier
router-fit rows as history, because this is information that would have
been available online.

However, router-validation rows may not influence:

- empirical-percentile references;
- robust centering/scaling values;
- q90/q95/q99 feature thresholds;
- router-feature StandardScaler parameters;
- TCN teacher thresholds;
- logistic-regression coefficients.

## 6. Offline temporal-evidence teacher

The frozen TCN is run offline on CAL.

Teacher thresholds are derived from the **router-fit subset only**.

High temporal evidence:

\[
y^{90}_t =
\mathbf{1}
[
s^{TCN}_t
\ge Q^{fit}_{0.90}
].
\]

High-severity TCN alert evidence:

\[
y^{995}_t =
\mathbf{1}
[
s^{TCN}_t
\ge Q^{fit}_{0.995}
].
\]

Both quantiles use `higher` interpolation.

The primary logistic target is \(y^{90}\).

TCN scores are never online router inputs.

## 7. Cheap causal router features

The following feature set is frozen.

### 7.1 PCA EWMA percentile

Take the frozen Robust-PCA causal-EWMA score and convert it to a
right-inclusive empirical percentile using router-fit CAL only.

### 7.2 PCA EWMA first difference

\[
\Delta_t = e_t-e_{t-1}.
\]

Center by the router-fit median and divide by router-fit IQR.

If IQR is zero, use scale 1.0.

### 7.3 Recent PCA volatility

Calculate the causal rolling 12-bin standard deviation of PCA causal-EWMA.

Convert it to a right-inclusive router-fit empirical percentile.

### 7.4 Feature-support percentile

Fit StandardScaler on dataset TRAIN only.

For each row calculate q0.95 of the absolute standardized model features.

Convert it to a router-fit empirical percentile.

### 7.5 PCA latent-support percentile

Using the frozen Robust-PCA retained coordinates:

\[
S_{\mathrm{latent},t}
=
\sqrt{
\sum_k
\frac{z_{tk}^2}
{\lambda_k}
}.
\]

Convert it to a router-fit empirical percentile.

### 7.6 Combined support

\[
S_t
=
\max(
P_{\mathrm{feature},t},
P_{\mathrm{latent},t}
).
\]

### 7.7 Recent q90 hit fraction

Fraction of the current and previous 11 PCA-EWMA rows at or above the
router-fit PCA q90 threshold.

### 7.8 Recent q95 hit fraction

Same definition using the router-fit PCA q95 threshold.

All features are causal and TCN-free online.

A final StandardScaler for the eight router features is fitted on the
pooled MetroPT-3 + MetroPT2 router-fit rows only.

## 8. Router model

The router family is logistic regression.

Frozen implementation:

- solver: `liblinear`;
- max iterations: 2000;
- random state: 20260915.

Grid:

\[
C\in\{0.01,0.1,1,10\}
\]

class weight:

- `None`;
- `balanced`.

Routing probability threshold:

- 0.05
- 0.10
- 0.20
- 0.30
- 0.40
- 0.50

Total candidates:

\[
4\times2\times6 = 48.
\]

No values may be added after development results are observed.

Each logistic model is fitted on the pooled router-fit rows from both
development datasets.

## 9. Validation eligibility

Every candidate is evaluated separately on the chronological
router-validation subset of both datasets.

Data adequacy requires at least 20 high-evidence validation bins in each
dataset.

If a validation dataset contains at least 3 q0.995 alert-evidence bins,
the alert constraint applies.

A candidate is eligible only if each development dataset satisfies:

\[
\text{high-evidence coverage}\ge0.80
\]

and, when applicable,

\[
\text{TCN-alert coverage}\ge0.80.
\]

## 10. Candidate selection

Among eligible candidates minimize:

\[
\frac{
r_{\mathrm{MP3}}
+
r_{\mathrm{MP2}}
}{2},
\]

where \(r\) is validation TCN invocation fraction.

Frozen tie breaks:

1. maximize worst-dataset TCN-alert coverage;
2. maximize worst-dataset high-evidence coverage;
3. prefer lower \(C\);
4. prefer higher routing-probability threshold;
5. prefer `class_weight=None`.

If no candidate is eligible, close this router family without rescue tuning.

## 11. Why this family can exceed fixed q90 coverage

The previous study imposed:

\[
R_{\mathrm{router}}\subseteq R_{q90}.
\]

This study does not.

For example, a window may have PCA below q90 but simultaneously show:

- high PCA percentile relative to its recent support;
- positive EWMA slope;
- elevated recent EWMA volatility;
- elevated support distance.

The logistic router may route that window even though fixed q90 does not.

Therefore 80% TCN-evidence coverage is no longer mathematically bounded by
fixed-q90 coverage.

Whether the cheap signals are predictive enough is now an empirical
development question.

## 12. Policy freeze

If a candidate is selected, freeze before any simulation or new-data
evaluation:

- the eight router feature definitions;
- all transform/reference parameters;
- final router-feature StandardScaler;
- logistic coefficients and intercept;
- C;
- class weight;
- probability threshold;
- development teacher thresholds;
- validation metrics.

## 13. Simulated mechanism tests

After policy freeze, the following SIMULATED stress tests are allowed:

- feature location shift;
- feature scale shift;
- support-tail inflation;
- operating-mixture shift.

They test mechanism behavior only.

They cannot satisfy a generalization claim.

## 14. Genuinely new data

A primary claim requires genuinely new chronological compressor data not
used anywhere in method design.

Before opening EVALUATION, freeze a new-data protocol containing:

- sensor/feature mapping;
- TRAIN/CALIBRATION/EVALUATION boundaries;
- missing-data handling;
- gap handling;
- feature construction;
- TCN training;
- PCA training;
- CAL-only feature-reference construction.

The frozen logistic router coefficients and probability threshold may not
be changed for the new dataset.

Dataset-specific unsupervised TRAIN/CAL transforms are permitted because
the eight router inputs were designed to be normalized, transferable
quantities.

## 15. Primary success criteria on new data

The evidence-aware router succeeds only if all are true:

\[
\text{high-TCN-evidence coverage}\ge80\%
\]

\[
\text{TCN-alert coverage}\ge80\%
\]

\[
\text{mean gross GPU-device TCN energy reduction}\ge50\%
\]

and gross GPU-energy reduction is positive in all five measured
repetitions.

No weighted score can compensate for a failed criterion.

## 16. Energy boundary

Reuse the previously calibrated GPU protocol:

- in-process NVML;
- 100 ms sampling;
- two warm-ups;
- five measured repetitions;
- paired 10-second idle;
- at least 20 seconds metered workload.

Measured scope:

**GPU-device TCN scoring energy only.**

Router CPU energy remains UNKNOWN unless a validated CPU hardware-energy
counter becomes available.

Whole-system energy is UNKNOWN.

Physical compressor energy is not measured.

## 17. Stop conditions

Stop or create another new preregistration if:

- either frozen TCN checkpoint identity fails;
- the 60/40 chronological split cannot be constructed;
- either validation set has fewer than 20 high-evidence bins;
- no logistic candidate satisfies the frozen validation constraints;
- a router feature needs to change after development results are observed;
- reused TEST is accidentally used for development;
- new EVALUATION data is accessed before its protocol is frozen.

## 18. Interpretation guardrails

Do not call TCN evidence physical fault truth.

Do not use incident labels to train the router.

Do not claim causal diagnosis.

Do not call simulated stress testing independent validation.

Do not claim whole-system energy from GPU-device measurements.

Do not claim physical compressor-energy or carbon savings.

Do not promote a router without genuinely new chronological evaluation.
