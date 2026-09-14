# Energy-Aware Hierarchical Intelligence — Primary Decision

## Status

**Primary MetroPT-3 exploratory benchmark completed.**

Evidence class: **EXPLORATORY_REUSED_TEST_BENCHMARK**

This result does **not** change the production detector, does **not** promote a new detector, and does **not** establish physical compressor-energy savings.

## Frozen benchmark identity

- Pre-TEST benchmark freeze commit: `3a4a00df1f35cf2f6b1c0f25568f035ce97a47a7`
- Frozen TCN checkpoint SHA256: `3a2a98fd65ae91014aba89b04b953b984d6763003351ec03724d682632bc8f60`
- Operational detector: frozen Robust-PCA
- Temporal evidence model: StandardScaler causal TCN
- Router thresholds: calibration-only PCA causal-EWMA q90 / q95 / q99
- GPU energy measurement: in-process NVML, 100 ms sampling
- Warm-ups: 2 per condition
- Measured repetitions: 5
- Minimum metered workload: 20 s
- Frozen materiality threshold: mean gross GPU-energy reduction >= 3%
- CPU energy: UNKNOWN
- Whole-system energy: UNKNOWN

## Frozen PCA reproduction

The primary energy benchmark executed only after the frozen Robust-PCA reproduction guard passed.

The benchmark therefore preserved the existing operational reference and did not reinterpret the TCN as an operational detector.

## Primary energy result

Measured quantity: **GPU-device energy for the TCN scoring workload only**.

| Router | TEST TCN invocation fraction | Mean gross GPU-energy reduction | Mean incremental GPU-energy reduction | Gross reduction directionally consistent |
|---|---:|---:|---:|---|
| q90 | 23.555% | 75.184% | 75.335% | Yes, 5/5 |
| q95 | 15.183% | 83.608% | 83.734% | Yes, 5/5 |
| q99 | 9.484% | 90.324% | 90.392% | Yes, 5/5 |

All three routing points exceeded the frozen 3% materiality threshold by a very large margin.

Gross reduction ranges across five repetitions were:

- q90: 74.346% to 75.781%
- q95: 82.867% to 83.899%
- q99: 89.396% to 90.776%

The corresponding sample standard deviations of the gross reduction fraction were approximately:

- q90: 0.601 percentage points
- q95: 0.425 percentage points
- q99: 0.564 percentage points

Gross and matched-idle-adjusted incremental reductions were closely aligned, so the observed reduction is not an artifact of the idle correction.

## Evidence-retention result

| Router | TCN alert-bin coverage | Incident-related alert-bin coverage | TCN episode coverage | Incident-related episode coverage |
|---|---:|---:|---:|---:|
| q90 | 71.536% | 84.266% | 79.785% | 84.211% |
| q95 | 68.213% | 84.266% | 69.767% | 84.211% |
| q99 | 62.744% | 83.829% | 51.699% | 78.947% |

The router is strongly selective for incident-relevant temporal evidence: q95 invokes the TCN on only 15.183% of valid TEST windows while retaining 84.266% of incident-related TCN alert bins and 84.211% of incident-related TCN alert episodes.

## Pareto interpretation

### q90

q90 is the most conservative routing point.

It preserves the largest fraction of all TCN evidence, but it uses substantially more temporal-model inference than q95 while providing no improvement over q95 in the two incident-related coverage metrics.

For the study's incident-centered objective, q90 is therefore not the strongest operating point.

### q95

q95 is the strongest balanced operating point in the primary benchmark.

Compared with q90, q95:

- reduces the TCN invocation fraction from 23.555% to 15.183%;
- increases mean gross GPU-energy reduction from 75.184% to 83.608%;
- retains exactly the same observed incident-related alert-bin coverage, 84.266%;
- retains exactly the same observed incident-related episode coverage, 84.211%;
- gives up some non-incident / overall evidence coverage.

For the stated research question, q95 is the preferred exploratory operating point.

### q99

q99 is the aggressive low-compute operating point.

It invokes the TCN on only 9.484% of valid TEST windows and reduces gross GPU-device energy by 90.324%.

Relative to q95, the additional energy reduction is about 6.72 percentage points, while:

- incident-related alert-bin coverage falls only from 84.266% to 83.829%;
- incident-related episode coverage falls from 84.211% to 78.947%;
- overall episode coverage falls much more sharply, from 69.767% to 51.699%.

q99 is therefore attractive when compute/energy pressure dominates, but it is less conservative about preserving temporal evidence.

## Calibration-to-TEST routing shift

The router operating points were named q90 / q95 / q99 because their thresholds were frozen from the CALIBRATION PCA causal-EWMA distribution.

On CALIBRATION, the routed fractions were approximately 10%, 5%, and 1%.

On TEST, the routed fractions became:

- q90: 23.555%
- q95: 15.183%
- q99: 9.484%

This is a meaningful distribution-shift result. The routing quantiles do **not** imply a fixed inference budget after deployment.

The system still saved substantial TCN GPU energy on this TEST distribution, but future compute budgets should be expected to depend on operating support.

## Relation between invocation reduction and GPU-energy reduction

The observed GPU-energy reduction closely tracks the reduction in TCN invocations:

- q90 skips 76.445% of TCN windows and saves 75.184% gross GPU energy;
- q95 skips 84.817% of TCN windows and saves 83.608%;
- q99 skips 90.516% of TCN windows and saves 90.324%.

This supports the intended hierarchical-compute mechanism: the energy reduction is primarily explained by avoiding unnecessary temporal-model inference rather than by a post-hoc accounting effect.

## Primary research decision

The primary research question receives an **exploratory YES**.

A frozen Robust-PCA router can selectively invoke the causal TCN and materially reduce measured GPU-device TCN inference energy while retaining a substantial majority of incident-relevant TCN evidence on reused MetroPT-3 TEST.

The strongest balanced point is **q95**:

- 15.183% TCN invocation fraction;
- 83.608% mean gross GPU-energy reduction;
- 84.266% incident-related alert-bin coverage;
- 84.211% incident-related episode coverage.

This is a research result, not a production promotion.

## What this result does not prove

This benchmark does not prove:

- whole-system compute-energy savings;
- CPU/router energy savings;
- physical compressor-energy savings;
- carbon savings;
- superiority of the TCN as an operational anomaly detector;
- causal diagnosis;
- independent generalization, because MetroPT-3 TEST is reused exploratory evidence.

PCA/router execution occurs on CPU outside the GPU meter. CPU energy is therefore UNKNOWN.

## Next research action

Proceed to the preregistered MetroPT2 transport benchmark **without retuning**:

- same TCN architecture;
- same StandardScaler rule;
- same router quantiles q90 / q95 / q99;
- same evidence definitions;
- same energy-accounting definitions;
- no MetroPT2-based parameter adjustment.

The MetroPT2 result should be treated as a previously-inspected cross-dataset transport stress test, not untouched independent confirmation.

If the energy/evidence trade-off collapses on MetroPT2, close or narrow the generalization claim rather than tuning the primary study after the fact.
