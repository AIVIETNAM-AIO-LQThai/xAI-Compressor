# External Dataset Selection — RCSD-1YD

## Decision

Select **RCSD-1YD: Refinery Compressor Sensor Data, One-Year Dataset**
as the new chronological external dataset for the evidence-aware compute-routing
study.

Dataset DOI:

`10.5281/zenodo.14866092`

Descriptor DOI:

`10.1109/IEEEDATA.2025.3571011`

The dataset is from a real centrifugal compressor operating in an oil refinery
in Greece. It covers 1 January 2022 through 31 December 2022, contains 25
process-sensor channels plus timestamp, is sampled every 15 minutes, and
contains approximately 35,036 readings per sensor.

The dataset has not been used to train, tune, select, or validate the frozen
evidence-aware router. Public metadata and the descriptor paper have been
reviewed only to determine suitability and freeze this protocol before data
inspection.

## Why RCSD-1YD is the preferred public candidate

It satisfies the most important requirements simultaneously:

- real industrial compressor rather than synthetic machinery;
- full chronological timestamps over one calendar year;
- multivariate process measurements;
- pressure, temperature, vibration, axial-displacement, and speed signals;
- public archival DOI;
- sufficiently long chronological blocks for TRAIN, CALIBRATION, and EVALUATION;
- different compressor type and industrial context from MetroPT;
- no dependence on fault labels for the primary router-evidence evaluation.

The descriptor reports 15-minute sampling and one known one-hour data gap
around the daylight-saving-time change on 27 March 2022.

The compressor is reported to be predominantly stable: 23 of 25 monitored
parameters stayed within their defined operating thresholds, while the speed
indicator and lube-oil header pressure showed threshold deviations. Those
threshold observations are not used to train or tune the evidence-aware router.

## Sensor scope frozen from public descriptor metadata

Use all 25 published process channels:

Axial displacement:
- 75ZI800BA.pv
- 75ZI800BB.pv
- 75ZI801BA.pv
- 75ZI801BB.pv

Pressure:
- 75PI808.pv
- 75PI823.pv
- 75PI870.pv
- 75PI845.pv
- 75PDI853.pv

Temperature:
- 75TI821.pv
- 75TI822.pv
- 75TI827.pv
- 75TI828.pv
- 75TI824.pv
- 75TI829.pv
- 75TI830.pv
- 75TI831.pv
- 75TI832.pv
- 75TI836.pv
- 75TI834.pv

Vibration:
- 75XI821BX.pv
- 75XI822BY.pv
- 75XI823BX.pv
- 75XI824BX.pv

Speed:
- 75SI865R.pv

No channel may be dropped after EVALUATION is inspected because of performance.
A channel may be excluded before model adaptation only if the downloaded file
fails schema validation for that channel; such a failure requires a documented
protocol amendment before EVALUATION metrics are produced.

## Candidates not selected

### MetroPT2

Public and highly compatible, but it has already been used extensively in this
project for transport development. It cannot serve as genuinely new external
evaluation.

### IITK air-compressor acoustic dataset

Real compressor data and public, but it is an acoustic seeded-fault benchmark
rather than a long multivariate chronological DCS stream, and it has already
been inspected in prior cross-dataset work.

### 2026 semiconductor-fab air-compressor study

Scientifically attractive: multi-year, multi-compressor, one-minute SCADA data.
However, the authors state that the data are confidential, so it cannot support
a reproducible public external evaluation.

### Cégep de Sept-Îles industrial compressor study

The paper describes real timestamped pressure, temperature, and electrical
current data from an industrial compressor and states that supporting data are
openly available. However, the public paper does not expose a comparably clear,
versioned archival dataset with the chronology and schema metadata needed for a
blind one-shot protocol. RCSD-1YD therefore has stronger reproducibility and
data-governance properties for the present evaluation.

### Synthetic predictive-maintenance datasets

Rejected because synthetic machine data cannot satisfy the requirement for a
genuinely new real compressor evaluation.

## Scope of the eventual claim

RCSD-1YD is a **cross-domain external transport evaluation**:

Metro train reciprocating/air-production compressor development
→ refinery centrifugal gas compressor evaluation.

Passing RCSD-1YD would support transport of the energy-aware routing mechanism
to a substantially different real compressor context. It would not establish
universal compressor generalization.
