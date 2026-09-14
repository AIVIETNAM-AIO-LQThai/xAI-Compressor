# Support-Aware Energy Routing — Calibration Decision

## Status

**DEVELOPMENT FAILURE — CLOSE THIS ROUTER FAMILY**

No TEST partition was opened for policy selection.

- MetroPT-3 TEST opened: false
- MetroPT2 TEST opened: false
- new evaluation data opened: false
- incident labels used for selection: false
- eligible candidates: 0 / 8
- selected policy: none

The study therefore stops at the preregistered development-stage failure condition.

## Why the failure is structural

The adaptive router can choose only one of three PCA thresholds:

- q90
- q95
- q99

with q90 the least restrictive threshold.

For every row,

\[
T_{\mathrm{adaptive},t} \in \{T_{q90}, T_{q95}, T_{q99}\}
\]

and

\[
T_{q90} \le T_{\mathrm{adaptive},t}.
\]

Therefore any adaptive routing set is a subset of the fixed-q90 routing set:

\[
R_{\mathrm{adaptive}} \subseteq R_{q90}.
\]

Consequently, adaptive evidence coverage cannot exceed fixed-q90 evidence coverage for the same evidence target.

The fixed-q90 calibration baselines were already far below the preregistered 80% high-evidence requirement:

| Dataset | q90 invocation | q90 high-evidence coverage | q90 TCN-alert coverage |
|---|---:|---:|---:|
| MetroPT-3 CAL | 10.265% | 20.690% | 88.889% |
| MetroPT2 CAL | 10.354% | 10.432% | 14.286% |

This makes the 80% high-evidence requirement mathematically unreachable for every router in the frozen q99/q95/q90 family.

MetroPT2 also makes the 80% TCN-alert requirement unreachable within this family because even fixed q90 covers only 14.286% of the CAL q0.995 TCN-alert bins.

The failure is therefore not primarily a poor choice of support boundaries. It is a mismatch between the admissible PCA-threshold routing family and the TCN evidence target.

## Candidate-grid result

All eight frozen support-boundary candidates failed.

The best observed high-evidence coverage remained below 20% on MetroPT-3 and below 7% on MetroPT2 for the adaptive candidates, while invocation fractions stayed around 1.4%–2.6%.

The support controller successfully changed compute expenditure, but the admissible route remained constrained to windows already admitted by fixed q90.

## Scientific conclusion

The following hypothesis is **not supported**:

> Support novelty alone can make a q99/q95/q90 PCA-threshold controller preserve at least 80% of TCN temporal evidence across both development datasets.

The completed result suggests a stronger distinction:

> Operating-support novelty and "the TCN is worth running here" are not the same signal.

A window may be in familiar support yet contain temporal structure that produces high TCN evidence. Conversely, an out-of-support window need not contain useful temporal evidence.

## What must not be changed in this study

Do not:

- lower the 80% evidence criterion;
- add new support-boundary values;
- add a q80 or looser PCA threshold;
- change max(feature-support, latent-support) after observing this result;
- use either reused TEST partition to rescue the router;
- promote any candidate.

Any such intervention requires a new preregistration.

## Next router family

The next study should decouple "route the TCN" from "PCA score exceeds q90".

A cheap evidence-aware escalation model should predict whether the TCN is likely to produce high temporal evidence using only cheap, causal, non-TCN inputs such as:

- PCA causal-EWMA score;
- PCA score percentile;
- EWMA first difference / slope;
- recent EWMA volatility;
- support percentile;
- feature-support percentile;
- PCA latent-support percentile;
- cheap persistence counters;
- operating-state / regime indicators.

The route should then be allowed to fire even when the PCA anomaly score is below q90.

A suitable new optimization problem is:

\[
\min \; \text{TCN invocation fraction}
\]

subject to

\[
\text{high-TCN-evidence coverage} \ge 0.80
\]

and

\[
\text{TCN-alert coverage} \ge 0.80.
\]

The router must be trained and selected on TRAIN/CALIBRATION only and evaluated once on genuinely new chronological data.

## Decision

**Close `research/support-aware-energy-routing` after recording this calibration failure.**

Start a separate preregistered study for an evidence-aware compute router.
