# CoBra Schema and Chronology Inspection Protocol

Status: **FREEZE BEFORE OPENING TEST-DAY CSV CONTENT**

Parent source-lock commit:

`9d3e535ae7e6932fce1a4ec975d0ebacdfa4b0f6`

Selected source:

- CoBra High Temperature Heat Pump Demonstrator — Experimental Dataset 2024–2025
- DOI `10.82481/cobra.2025`
- frozen source version `1.0.0`
- 23 dated test-day archives
- `overview_experiments.csv`

Evidence role:

`FUTURE_GENUINELY_INDEPENDENT_EXTERNAL_VALIDATION`

## Purpose

Inspect only the source structure needed to make CoBra reproducible for later
Router V2 validation:

- ZIP member names;
- CSV header / delimiter;
- timestamp-column identity and parseability;
- row count;
- first and last timestamp;
- chronological monotonicity;
- duplicate timestamp count;
- timestamp-step metadata;
- per-row field-count consistency;
- column names and count;
- coarse dtype class inferred without computing sensor distributions.

This stage must not compute any model, anomaly, feature-distribution, target,
coverage, or energy outcome.

## Frozen whole-test-day split

The split is frozen **before inspecting sensor behavior** and depends only on
the already-known chronological test-day filenames.

Sort the 23 dated test days ascending by date and assign:

- first 14 test days → `TRAIN`
- next 4 test days → `CALIBRATION`
- final 5 test days → `EVALUATION`

This is a whole-day split. No test day may be divided across partitions.

Frozen dates:

### TRAIN

1. 2024-06-03
2. 2024-06-06
3. 2024-06-10
4. 2024-06-12
5. 2024-06-17
6. 2024-06-19
7. 2024-06-24
8. 2024-06-27
9. 2024-11-11
10. 2024-11-21
11. 2024-11-25
12. 2024-12-11
13. 2024-12-18
14. 2025-01-15

### CALIBRATION

15. 2025-01-17
16. 2025-01-20
17. 2025-01-22
18. 2025-02-06

### EVALUATION — BLIND FUTURE TEST

19. 2025-02-17
20. 2025-02-18
21. 2025-02-20
22. 2025-02-24
23. 2025-02-25

The split may not be changed because of row count, missingness, operating mode,
model score, or later performance.

## Allowed schema inspection

For all partitions, including EVALUATION, the inspection implementation may:

- list ZIP members without extracting them;
- identify the single primary test-day CSV member;
- read the CSV header;
- identify the timestamp column;
- stream rows for:
  - row count;
  - timestamp parsing only;
  - first/last timestamp;
  - timestamp monotonicity;
  - duplicate timestamps;
  - timestamp-step counts;
  - field-count consistency;
  - whether each non-timestamp field is parseable as numeric, boolean, or text,
    reported only as a column-level schema class.

For non-timestamp sensor fields, the implementation may not retain, print,
serialize, aggregate, rank, correlate, or otherwise expose actual values.

The implementation should process EVALUATION sensor fields only to the minimum
extent required to determine schema class and row structural validity.

## Allowed source metadata

`overview_experiments.csv` may be read as source metadata after this protocol is
frozen.

Its descriptive experiment names / notes may be recorded, but they may not be
used to modify the frozen split or to select Router V2 features/hyperparameters.

## Explicitly forbidden

This stage must not compute or report:

- sensor min/max/mean/median/quantiles;
- missing-value rate by sensor if determining it requires outcome-like value
  inspection beyond structural blank/nonblank schema checks;
- sensor correlation;
- PCA or reconstruction error;
- EWMA evidence;
- support percentiles;
- TCN scores or temporal prediction loss;
- Router V1 or Router V2 probabilities;
- high-evidence prevalence;
- alert prevalence;
- threshold coverage;
- anomaly/fault rate;
- energy measurements;
- any comparison of TRAIN/CAL/EVALUATION sensor values.

Do not plot sensor values.

Do not open EVALUATION in pandas or another dataframe library for exploratory
analysis.

## Structural missingness rule

The inspection may count rows containing structurally blank CSV fields only for
the purpose of determining whether a deterministic future ingestion rule is
needed.

If blank-field counts are reported, they must be:

- row-structural counts only;
- not used to select sensor subsets;
- not used to alter the split;
- not interpreted as anomalies or failures.

No imputation rule is selected in this stage unless it follows mechanically from
the frozen schema result and is frozen in a later adaptation protocol.

## Timestamp-step reporting

The timestamp analysis may report counts of exact time differences between
consecutive timestamps because sampling regularity is required to design causal
windows.

This is chronology metadata, not an outcome metric.

No sensor-dependent gap detection is allowed.

## Inspection output

The next implementation may generate:

`docs/research/cobra_schema_chronology_manifest.json`

The manifest must include:

- source-lock commit;
- source hashes from the committed source-lock manifest;
- split assignment by date;
- per-archive ZIP member inventory;
- primary CSV member;
- column names;
- timestamp column;
- total row count;
- first and last timestamp;
- monotonic / duplicate status;
- timestamp-step counts;
- row field-count consistency;
- coarse schema classes;
- structural blank-row counts if present;
- explicit flags that model/outcome metrics remain unopened.

## Stop conditions

Stop before Router V2 development if any of the following occurs:

- a source hash differs from the committed source-lock manifest;
- a dated archive contains no unambiguous primary CSV;
- timestamp parsing cannot be made deterministic;
- chronological order within a test day is ambiguous;
- EVALUATION sensor values are accidentally summarized or used for a model
  outcome.

A structural mismatch must be documented; it must not be silently repaired.

## After this inspection

Once the schema/chronology manifest is reviewed and committed, freeze a
dataset-specific CoBra adaptation/evaluation protocol.

Only after that future-EVALUATION boundary is fully frozen may Router V2
development begin on development data.

CoBra EVALUATION remains unavailable for Router V2 model selection until the
final one-shot external validation.
