# Energy-Aware Hierarchical Intelligence — MetroPT2 Transport Decision

## Status

**MetroPT2 transport stress test completed.**

Evidence class: **PREVIOUSLY_INSPECTED_CROSS_DATASET_TRANSPORT**

This is not untouched independent confirmation. MetroPT2 had already been used in earlier repository research.

Primary MetroPT-3 result commit:
`de4dfec05167358d385c3fd21e12f195135db3fd`

MetroPT2 TCN checkpoint SHA256:
`8709a5ce5e08225b2573e95960086c1e7641c31f752f656b9e50c0b23a7e2bf0`

No MetroPT2 TEST tuning was performed.

## Integrity checks

The transport benchmark preserved the frozen study design:

- MetroPT2 `flowmeter_excluded` feature variant
- 63 features
- exact primary feature-name/order identity
- StandardScaler for the TCN, fit on MetroPT2 TRAIN only
- same causal TCN architecture and training hyperparameters
- same raw-MSE TCN evidence definition
- same no-smoothing / no-persistence TCN evidence semantics
- same q90 / q95 / q99 router quantile definitions
- numeric router thresholds estimated from MetroPT2 CALIBRATION only
- same GPU-energy measurement protocol
- same 3% materiality threshold

The previously committed MetroPT2 flowmeter-excluded Robust-PCA benchmark reproduced successfully.

## TCN conditioning

The MetroPT2 TCN was numerically well conditioned:

- best epoch: 34
- epochs run: 39
- best validation MSE: 0.3639505556
- CAL q0.995 anomaly threshold: 3.5982227325
- CAL top-3 feature MSE share: 12.786%

This is close to the primary MetroPT-3 StandardScaler TCN conditioning:

- primary best validation MSE: 0.3520388965
- primary CAL top-3 MSE share: 14.457%

Therefore the transport degradation is not explained by the earlier RobustScaler/MSE conditioning pathology.

## Router calibration and TEST invocation

Calibration fractions were approximately the intended quantiles:

| Router | CAL routed fraction | TEST TCN invocation fraction |
|---|---:|---:|
| q90 | 10.007% | 12.528% |
| q95 | 5.003% | 5.708% |
| q99 | 1.008% | 2.018% |

Unlike MetroPT-3, MetroPT2 TEST did not show a very large routing-budget shift. q90 and q95 remained relatively close to their calibration fractions, while q99 approximately doubled from 1.0% to 2.0%.

## MetroPT2 energy result

Measured quantity: **GPU-device energy for TCN scoring only**.

| Router | TEST TCN invocation | Mean gross GPU-energy reduction | Mean incremental reduction | Gross sample SD |
|---|---:|---:|---:|---:|
| q90 | 12.528% | 86.978% | 87.666% | 0.461 pp |
| q95 | 5.708% | 93.634% | 93.744% | 0.321 pp |
| q99 | 2.018% | 95.829% | 96.516% | 0.122 pp |

All three operating points exceeded the frozen 3% materiality threshold in all five measured repetitions.

The compute mechanism transported strongly:

- q90 skips 87.472% of TCN windows and saves 86.978% gross GPU energy;
- q95 skips 94.292% of TCN windows and saves 93.634%;
- q99 skips 97.982% of TCN windows and saves 95.829%.

GPU-energy reduction therefore continues to track avoided TCN inference.

## MetroPT2 evidence-retention result

| Router | Alert-bin coverage | Incident-related alert-bin coverage | Episode coverage | Incident-related episode coverage |
|---|---:|---:|---:|---:|
| q90 | 72.425% | 74.591% | 24.658% | 57.143% |
| q95 | 32.189% | 28.805% | 15.068% | 57.143% |
| q99 | 17.918% | 13.082% | 13.699% | 57.143% |

There were 932 always-on TCN alert bins, of which 795 were incident-related, and 73 alert episodes, of which 7 were incident-related.

All three router thresholds covered 4 of the 7 incident-related TCN episodes. Because the denominator is only seven episodes, this episode metric is coarse. The incident-related alert-bin metric provides substantially more resolution.

## Cross-dataset comparison

### q90

MetroPT-3:
- TCN invocation: 23.555%
- gross GPU-energy reduction: 75.184%
- incident alert-bin coverage: 84.266%
- incident episode coverage: 84.211%

MetroPT2:
- TCN invocation: 12.528%
- gross GPU-energy reduction: 86.978%
- incident alert-bin coverage: 74.591%
- incident episode coverage: 57.143%

Transport interpretation:
q90 preserves a substantial majority of incident-related alert bins on both datasets, but episode coverage drops materially on MetroPT2.

### q95

MetroPT-3:
- TCN invocation: 15.183%
- gross GPU-energy reduction: 83.608%
- incident alert-bin coverage: 84.266%
- incident episode coverage: 84.211%

MetroPT2:
- TCN invocation: 5.708%
- gross GPU-energy reduction: 93.634%
- incident alert-bin coverage: 28.805%
- incident episode coverage: 57.143%

Transport interpretation:
the primary-study q95 balance does **not** transport. Its incident-related alert-bin coverage drops by about 55.46 percentage points.

### q99

MetroPT-3:
- TCN invocation: 9.484%
- gross GPU-energy reduction: 90.324%
- incident alert-bin coverage: 83.829%
- incident episode coverage: 78.947%

MetroPT2:
- TCN invocation: 2.018%
- gross GPU-energy reduction: 95.829%
- incident alert-bin coverage: 13.082%
- incident episode coverage: 57.143%

Transport interpretation:
q99 transports the compute-saving mechanism but not the evidence-retention behavior.

## Scientific interpretation

The transport result is **PARTIAL TRANSPORT**, not full replication and not total collapse.

What transported strongly:

1. the StandardScaler TCN remained numerically well conditioned;
2. calibration-only PCA routing produced controlled selective inference;
3. fewer TCN invocations produced large and repeatable GPU-energy reductions;
4. the energy-saving mechanism remained monotonic from q90 to q99.

What did not transport:

1. the MetroPT-3 q95 energy/evidence balance;
2. high incident-related evidence retention at q95 and q99;
3. high incident-related episode coverage.

The most likely descriptive conclusion is that **compute selectivity transports more robustly than evidence alignment**. The PCA router and TCN evidence model do not maintain the same cross-dataset alignment.

This is not a causal explanation for why the alignment changes.

## Decision

The original broad claim should be narrowed.

Supported:

> A cheap Robust-PCA gate can materially reduce GPU-device temporal-model inference energy across both MetroPT-3 and MetroPT2 under frozen calibration-only routing.

Supported with caution:

> Conservative routing can retain a substantial fraction of incident-related temporal evidence, but the retained fraction is dataset dependent.

Not supported:

> q95 is a generally robust balanced operating point across MetroPT datasets.

Not supported:

> calibration quantile alone determines a portable energy/evidence trade-off.

The primary MetroPT-3 q95 conclusion remains valid for that exploratory dataset, but it should not be promoted as a cross-dataset default.

## Post-hoc observation, not promotion

Within MetroPT2, q90 is the least aggressive and most evidence-preserving tested point:

- 12.528% TCN invocation
- 86.978% gross GPU-energy reduction
- 74.591% incident-related alert-bin coverage
- 57.143% incident-related episode coverage

However, selecting q90 as a new global default after seeing MetroPT2 TEST would be post-hoc model selection.

Therefore q90 may be described as the strongest MetroPT2 transport point among the frozen candidates, but it must **not** be promoted without a new preregistered study and genuinely new evaluation data.

## What this result does not prove

This benchmark does not prove:

- independent generalization;
- whole-system compute-energy savings;
- CPU/router energy savings;
- physical compressor-energy savings;
- carbon savings;
- causal fault diagnosis;
- superiority of the TCN as an operational detector.

CPU/router and whole-system energy remain UNKNOWN.

## Recommended next research question

Do not tune q90/q95/q99 further on MetroPT-3 or MetroPT2 TEST.

A new study should instead ask:

> Can a calibration-only router adapt its compute budget to support shift while preserving a preregistered minimum level of temporal evidence on genuinely new chronological compressor data?

Candidate directions for a new preregistered study include:

- support-aware routing using only TRAIN/CALIBRATION diagnostics;
- uncertainty-aware escalation;
- evidence-coverage constraints estimated without TEST labels;
- a budget controller that treats routing rate as a deployment resource constraint;
- new independent chronological compressor data for confirmation.

Any such work should start from a new protocol rather than modifying the completed transport benchmark.
