# Support-Aware Energy Routing

Status: **PREREGISTRATION — no new router experiment executed**

Branch:

`research/support-aware-energy-routing`

Frozen parent commit:

`98ab0c74a1603f4def72b962a596282ff60a3b6b`

## 1. Why this study exists

The completed energy-aware hierarchical study produced two findings.

First, selective TCN execution substantially reduced measured GPU-device
TCN scoring energy on both MetroPT-3 and MetroPT2.

Second, the evidence-retention trade-off was not stable across those
datasets. In particular, the q95 operating point that was well balanced on
MetroPT-3 did not preserve comparable incident-related TCN evidence on
MetroPT2.

The follow-up study therefore does **not** ask which fixed q90/q95/q99
threshold is best.

It asks whether a cheap router can recognize operating-support shift and
adapt how much TCN compute it spends without using TEST labels or TEST
performance for policy selection.

## 2. Research question

> Can a calibration-only support-aware router adapt temporal-model compute
> expenditure under operating-support shift while preserving a
> preregistered minimum level of temporal-model evidence on genuinely new
> chronological compressor data?

## 3. Evidence status

The completed MetroPT-3 and MetroPT2 TEST partitions are reused,
previously inspected evidence.

They may motivate this protocol, but they are excluded from router-policy
development and selection.

Policy development uses only:

- MetroPT-3 TRAIN and CALIBRATION;
- MetroPT2 TRAIN and CALIBRATION.

The primary generalization claim requires genuinely new chronological
compressor data that has not previously been used for model, router,
threshold, feature, or policy selection in this project.

If such a dataset is unavailable, the study may proceed through calibration,
preflight, and clearly labeled simulated stress tests, but it must stop
before making a primary generalization claim.

## 4. Frozen architecture

The operational detector remains the frozen Robust-PCA pipeline.

The temporal model remains a second-stage research evidence model.

The TCN does not change the operational alert.

Existing checkpoint identities used for development:

MetroPT-3:

`3a2a98fd65ae91014aba89b04b953b984d6763003351ec03724d682632bc8f60`

MetroPT2:

`8709a5ce5e08225b2573e95960086c1e7641c31f752f656b9e50c0b23a7e2bf0`

No architecture search is permitted in this study.

## 5. Cheap support signal

The online support signal must not require a TCN invocation.

Two cheap components are frozen.

### 5.1 Feature-support statistic

Fit a StandardScaler on TRAIN.

For each causal feature vector, calculate the absolute standardized value
of all model features and use the 95th percentile across features.

This provides a robust row-level summary of how far the current feature
vector lies from TRAIN location/scale.

### 5.2 PCA latent-support statistic

Use the already required Robust-PCA transform.

For retained PCA component scores z_k with TRAIN explained variance
lambda_k, calculate

\[
S_{\mathrm{latent}}
=
\sqrt{
\sum_k \frac{z_k^2}{\lambda_k}
}.
\]

This is a cheap latent-space support-distance statistic.

### 5.3 Calibration normalization

Each component is converted to an empirical percentile using the official
CALIBRATION distribution only.

The combined support score is

\[
S_t =
\max(
P_{\mathrm{CAL}}(S_{\mathrm{feature},t}),
P_{\mathrm{CAL}}(S_{\mathrm{latent},t})
).
\]

No incident labels are used.

Support score is evidence of distribution/support shift, not diagnosis of
a physical fault.

## 6. Support-aware compute rule

The PCA anomaly-routing score remains the frozen causal-EWMA PCA score.

For each development dataset, derive q90, q95 and q99 numeric PCA thresholds
from its official CALIBRATION split using `higher` interpolation.

The adaptive policy has three compute states:

- low support shift -> use q99;
- intermediate support shift -> use q95;
- high support shift -> use q90.

Thus the system spends more TCN compute when the current operating point is
farther from calibration support.

## 7. Calibration-only policy selection

A small candidate grid is frozen before execution.

Mid-support boundary:

- 0.80
- 0.90
- 0.95

High-support boundary:

- 0.95
- 0.975
- 0.99

Only candidates with mid < high are valid.

No additional values may be introduced after results are observed.

### Offline temporal-evidence teacher

The TCN may be run offline on CALIBRATION for router-policy calibration.

Two evidence targets are defined:

- high-evidence windows: raw TCN score >= CAL q0.90;
- TCN anomaly-alert windows: raw TCN score >= CAL q0.995.

Both use `higher` interpolation.

A candidate policy is eligible only if, on **each** development calibration
dataset:

1. it covers at least 80% of q0.90 high-evidence windows; and
2. it covers at least 80% of q0.995 TCN alert windows when at least five
   such alert windows exist.

Among eligible policies, select the one with the lowest mean TCN invocation
fraction across MetroPT-3 CALIBRATION and MetroPT2 CALIBRATION.

Tie breaking is frozen:

1. highest worst-dataset q0.995 TCN alert-bin coverage;
2. higher high-support boundary;
3. higher mid-support boundary.

After selection, the policy is frozen.

## 8. Baselines

The new evaluation must include:

- always-on TCN;
- fixed q90;
- fixed q95;
- fixed q99;
- selected support-aware router.

Fixed q90/q95/q99 remain calibration-only numeric thresholds for the new
dataset. The quantile definitions, not old dataset-specific numeric values,
are transported.

## 9. Primary success criteria

On genuinely new chronological compressor data, the support-aware router
must satisfy all of the following:

- >= 80% coverage of CAL-defined q0.90 TCN high-evidence windows;
- >= 80% coverage of CAL-defined q0.995 TCN alert bins;
- >= 50% mean gross measured GPU-device TCN scoring energy reduction versus
  always-on TCN;
- positive gross energy reduction in every measured repetition.

The 80% evidence target and 50% energy target are new-study preregistered
targets informed by the completed exploratory studies. They are not claimed
to be universal standards.

Incident-related evidence is secondary because genuinely new chronological
data may not contain trusted incident annotations.

If incident labels are available, report incident-related TCN alert-bin and
episode coverage without changing the primary criteria.

## 10. Energy measurement

Reuse the completed measurement protocol:

- NVIDIA GPU device power via in-process NVML;
- 100 ms sampling;
- two warm-ups per condition;
- five measured repetitions;
- paired 10-second idle measurement;
- minimum 20-second metered workload;
- gross and matched-idle-adjusted incremental energy reported separately.

The primary energy quantity remains GPU-device energy for TCN scoring.

CPU/router energy is UNKNOWN.

Whole-system energy is UNKNOWN.

Physical compressor energy is not measured.

## 11. Simulated support-shift stress tests

Before genuinely new real data is available, simulated support-shift tests
are allowed for mechanism validation only.

Allowed manipulations:

- feature location shift;
- feature scale shift;
- support-tail inflation;
- operating-mixture shift.

Every such result must be labeled `SIMULATED`.

A simulated result cannot satisfy the primary success criteria and cannot
be used to claim generalization.

## 12. Stop conditions

Stop the study or start a new preregistration if any of the following occurs:

- no candidate in the frozen calibration grid satisfies the 80% evidence
  constraints on both development datasets;
- a checkpoint identity does not reproduce;
- feature identity required by a development dataset does not reproduce;
- TEST data is accidentally used for policy selection;
- support-score or policy definitions need to change after evaluation;
- the new evaluation dataset has already been used for method selection.

## 13. Interpretation guardrails

Do not claim that support shift is a fault.

Do not claim that router selection diagnoses root cause.

Do not claim physical compressor-energy savings.

Do not claim carbon savings from GPU-device measurements alone.

Do not promote a support-aware router from MetroPT-3 or MetroPT2 reused
TEST performance.

Do not call simulated stress testing independent validation.

## 14. Planned sequence

1. Freeze this protocol.
2. Implement support-score calculation.
3. Reproduce both frozen development checkpoint identities.
4. Calibrate q90/q95/q99 and support percentiles on TRAIN/CALIBRATION only.
5. Evaluate the frozen candidate grid on both CALIBRATION sets.
6. Select exactly one support-aware policy under the preregistered rule.
7. Freeze the selected policy and preflight.
8. Run simulated stress tests only as mechanism diagnostics if useful.
9. Acquire and document genuinely new chronological compressor data.
10. Freeze its preprocessing/split before evaluation.
11. Evaluate fixed baselines and the support-aware router once.
12. Measure GPU-device TCN inference energy using the frozen protocol.
13. Record the decision without post-evaluation retuning.
