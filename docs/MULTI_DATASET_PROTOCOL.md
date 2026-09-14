# AeroXAI Multi-Dataset Validation Protocol

Status: **research branch only**

This protocol tests whether AeroXAI's current explanation concentration is specific to the MetroPT-3 air-leak incidents, or is a systematic consequence of the detector / feature representation.

It must not alter the frozen `esic-2026-v3` release.

## Research question

Does the distribution of operator-level anomaly evidence change across recording periods and failure types?

Primary explanation-distribution metrics:

```text
C1     = top-1 contribution share
C3     = sum of the three largest contribution shares
H      = Shannon entropy of normalized contribution shares
N_eff  = exp(H), the effective number of equally contributing groups
HHI    = sum(p_i^2)
```

These metrics quantify detector-evidence concentration. They do **not** measure the number of physical root causes.

## Datasets

### MetroPT-3

Role:

```text
frozen baseline
```

Use the already verified report:

```text
docs/xai_verification_report.json
```

Do not retrain or retune MetroPT-3 for this baseline.

### MetroPT2

Source:

```text
Zenodo DOI 10.5281/zenodo.7766691
MetroPT2.csv
```

Published metadata:

```text
7,116,940 rows
21 attributes
1 Hz logging
2022-04-28 to 2022-07-28
```

Reported failures:

```text
Air leak:
2022-06-04 10:19:24.300
through 2022-06-04 14:22:39.188

Oil leak:
2022-07-11 10:10:18.948
through 2022-07-14 10:22:08.046
```

The first MetroPT2 checkpoint is **data audit only**. No detector performance should be inspected until the audit is recorded.

## Pre-registered comparison principle

After the data audit passes, the MetroPT2 detector benchmark should preserve the frozen AeroXAI alert protocol wherever the dataset supports it:

```text
5-minute causal bins
RobustScaler + PCA
95% variance retained
calibration-only threshold
threshold quantile 0.995
EWMA alpha 0.20
persistence 3 / 4
10-minute smoothing reset gap
30-minute alert-episode merge
24-hour early-warning window
2-hour late tolerance
```

Train / calibration / test boundaries must be chronological and must place both published failures only in the final test partition.

Any adjustment to exact boundaries due to observed data gaps must be made from the data-quality audit only, before detector results are inspected.

## Flowmeter ablation

MetroPT2 includes a Flowmeter signal that is not available in MetroPT-3.

Run two separately labeled MetroPT2 experiments:

```text
A. full common physical sensor set + Flowmeter
B. identical protocol with Flowmeter excluded
```

Reason:

A 2026 MetroPT2 interpretability study reported very strong alignment between Flowmeter excursions and both documented failures and argued that the dataset becomes substantially more challenging without Flowmeter.

Therefore the Flowmeter-inclusive result must not be the only generalization result.

## Required outputs

For each MetroPT2 failure and each pre-registered sensor variant report:

```text
event timing
timely / anytime detection
pre-onset or post-onset timing
false alert episodes / 24 h
time in alert
PR-AUC
relevant episode precision

dominant explanation group
C1
C3
H
N_eff
HHI
```

Also report the overlap of top explanation groups between:

```text
MetroPT-3 air leaks
MetroPT2 air leak
MetroPT2 oil leak
```

## Interpretation rules

Allowed:

> The explanation evidence is more concentrated / more distributed under this detector and dataset.

Allowed:

> Different failure reports are associated with different dominant detector-evidence groups.

Not allowed:

> A higher effective group count means the physical fault is more complex.

Not allowed:

> The dominant group is the physical root cause.

Not allowed:

> MetroPT2 is an independent factory validation.

MetroPT-3 and MetroPT2 are closely related Porto Metro compressor benchmarks. They improve time-period and fault-type breadth, but do not establish cross-industry or cross-asset generalization.

## Promotion rule

No new dataset or model replaces the frozen V3 primary detector unless:

1. its protocol is fixed before test results are inspected;
2. its alert burden is operationally acceptable;
3. its ranking quality and event metrics justify promotion;
4. the result survives a genuinely independent validation set;
5. evidence provenance and explanation boundaries remain explicit.
