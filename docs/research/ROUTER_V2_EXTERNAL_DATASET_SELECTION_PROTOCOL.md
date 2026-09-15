# Router V2 Independent External Dataset Selection Protocol

Status: **FREEZE BEFORE ROUTER V2 DEVELOPMENT**

Parent failure-analysis commit:

`87e4d667b94ecdca0ce0a9499347e744cc48aa71`

Planned branch:

`research/router-v2-external-dataset-selection`

## Purpose

The RCSD-1YD external evaluation is closed as `EXTERNAL_TRANSPORT_FAIL`, and
its post-hoc failure analysis is complete.

Before developing Router V2, select and lock a third, genuinely independent
compressor dataset for later primary external validation.

The dataset-selection decision must be made without inspecting any Router V2
performance on the candidate data.

## Independence requirement

The selected dataset must not have influenced:

- Router V1 development;
- RCSD failure analysis;
- Router V2 feature design;
- Router V2 model family selection;
- Router V2 hyperparameter selection;
- Router V2 threshold selection.

Once selected, its future EVALUATION partition must remain unopened until
Router V2 is fully frozen.

## Eligibility criteria

A candidate is eligible only if all of the following hold:

1. Real measured data. Synthetic or simulation-only datasets are ineligible for
   the primary external-validation role.
2. The physical system contains a compressor as a central monitored component.
3. Multivariate sensor measurements are available.
4. Chronology is available through timestamps or an unambiguous acquisition
   order.
5. There are enough sequential observations to form causal temporal windows and
   a nontrivial chronological TRAIN/CAL/EVALUATION split.
6. At least five usable numeric sensor channels are available before any
   outcome-based filtering.
7. The source is publicly accessible from a stable repository, institutional
   archive, or documented public dataset source.
8. The dataset is not MetroPT-3, MetroPT2, RCSD-1YD, or the already-used IITK
   acoustic compressor dataset.
9. No Router V2 outcome metric has been computed on it before selection.
10. The selected dataset's exact source version and cryptographic hash can be
    recorded before model adaptation.

## Preference ordering

Among eligible datasets, prefer in this order:

1. real field/industrial operational telemetry;
2. real controlled industrial/test-facility compressor telemetry;
3. real laboratory compressor telemetry.

Prefer multivariate process signals such as pressure, temperature, current,
speed, vibration, flow, valve state, or displacement over datasets containing
only precomputed features.

Prefer longer, contiguous chronological coverage over small snapshot tables.

## Metadata allowed before selection

Allowed:

- dataset title;
- authors/institution;
- DOI or permanent source;
- license;
- compressor type;
- acquisition setting;
- listed sensor names/types;
- nominal sampling rate;
- stated row count/duration;
- file names and file sizes;
- source-provided checksums;
- documented missing-data structure if explicitly described by the publisher.

## Data inspection forbidden before selection

Before the candidate is selected and recorded, do not compute:

- sensor distributions;
- correlations;
- anomaly rates;
- PCA scores;
- TCN scores;
- Router V1 or Router V2 outputs;
- train/test performance;
- target prevalence;
- threshold coverage;
- GPU energy metrics.

Opening a file only to verify schema, timestamp parsing, row count, source
checksum, and deterministic chronology is allowed after the candidate has been
selected in principle, but before any model outcome is computed.

## Selection record

The selection decision must record:

- selected dataset;
- permanent source/DOI;
- why it is eligible;
- why it is independent;
- compressor type and acquisition setting;
- available sensor modalities;
- chronology/duration/sampling description;
- license/access status;
- known limitations;
- rejected alternatives and non-outcome reasons for rejection.

## After selection

After the dataset is selected:

1. download or archive the exact source version;
2. record MD5/SHA256;
3. freeze a chronological TRAIN/CAL/EVALUATION split;
4. create a dataset-specific adaptation protocol;
5. keep EVALUATION blind;
6. only then begin Router V2 development on MetroPT + RCSD development data.

Router V2 development may use RCSD-1YD EVALUATION because its independent-test
role has already been consumed, but the newly selected dataset may not be used
for Router V2 model selection.

## Interpretation

Selecting the dataset before Router V2 development prevents choosing an
external test set because it happens to favor the eventual model.

The selected dataset is a future validation resource, not development data.
