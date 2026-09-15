# CoBra Schema and Chronology Review

Status: **STRUCTURAL PASS — READY FOR ADAPTATION PROTOCOL**

This review closes the schema/chronology inspection stage for the frozen
CoBra 2024–2025 source.

No PCA, EWMA, support, TCN, router, target, coverage, anomaly, energy,
or other model-outcome metric was computed during this stage.

## Frozen split

The preregistered whole-day split remains unchanged:

- TRAIN: 14 days
- CALIBRATION: 4 days
- EVALUATION: 5 days

No day was moved or excluded because of schema, timestamp resolution,
row count, missing channels, or any model result.

## Structural readiness

All 23 dated archives were inspected.

Final inspection result:

- timestamp parsing is deterministic for all archives;
- all archives are monotonic non-decreasing in published timestamp order;
- all rows have structurally consistent CSV field counts;
- the primary CSV member is unambiguous for every archive;
- no structural blank rows were observed;
- the source is ready for a separately frozen adaptation protocol.

Partition row counts:

- TRAIN: 310869
- CALIBRATION: 98161
- EVALUATION: 125826

## Timestamp resolution

Most archives publish second-or-finer timestamps.

Three archives publish only minute-resolution timestamps:

- `CoBra20240627.zip` — TRAIN
- `CoBra20241121.zip` — TRAIN
- `CoBra20250120.zip` — CALIBRATION

Their source rows remain in monotonic contiguous minute blocks.

Repeated parsed timestamps in those files are expected consequences of
the published minute resolution:

- `CoBra20240627.zip`: 9042 repeated timestamp transitions
- `CoBra20241121.zip`: 19957 repeated timestamp transitions
- `CoBra20250120.zip`: 20999 repeated timestamp transitions

No artificial seconds were generated.

Source row order is preserved.

Observed source chronology gaps remain unchanged and are not
interpolated or repaired.

## Schema evolution

The largest observed schema contains 267 columns.

No archive contains a column outside this 267-column union.

The intersection common to all 23 test days contains 258 columns.

The eight June 2024 TRAIN archives contain 264 columns and lack:

- `BPB202-[Pa_Absolute]`
- `BPB203-[Pa_Absolute]`
- `BTB203-[°C]`

The three November 2024 TRAIN archives contain 266 columns and lack:

- `BTB203-[°C]`

From December 2024 onward the normal late schema contains 267 columns,
except for `CoBra20250224.zip`.

`CoBra20250224.zip` is an EVALUATION archive with 261 columns and lacks
six BSH006 vibration channels:

- `BSH006-A-acceleration-[m/s^2]`
- `BSH006-A-velocity-[mm/s]`
- `BSH006-B-acceleration-[m/s^2]`
- `BSH006-B-velocity-[mm/s]`
- `BSH006-C-acceleration-[m/s^2]`
- `BSH006-C-velocity-[mm/s]`

No extra replacement columns appear on that day.

## Ordered-schema implication

Although several early archives contain the same set of 264 column
names, their ordered schema hashes differ.

Therefore future ingestion must align columns by explicit column name.

Positional feature alignment is prohibited.

The same requirement applies to all future TRAIN, CALIBRATION, and
EVALUATION processing.

## Adaptation-protocol requirement

This stage does not select the final model feature subset.

The next CoBra adaptation/evaluation protocol must freeze, before model
outcomes are opened:

1. name-based schema alignment;
2. the allowable model feature set;
3. treatment of fields unavailable on some source days;
4. treatment of minute-resolution source timestamps;
5. causal window construction;
6. handling of genuine source chronology gaps;
7. the behavior when a required feature is structurally unavailable.

No rule may be selected using EVALUATION model performance.

A mechanically defined universal schema of 258 fields is available as a
schema-compatibility option, but feature selection is deferred to the
adaptation protocol.

## Independence boundary

EVALUATION inspection in this stage was structural only.

EVALUATION sensor distributions, model scores, anomaly prevalence,
teacher evidence, routing coverage, energy measurements, and model
performance remain unopened.

CoBra retains its role as future genuinely independent external
validation for Router V2.
