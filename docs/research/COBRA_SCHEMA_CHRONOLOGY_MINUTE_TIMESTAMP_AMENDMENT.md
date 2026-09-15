# CoBra Minute-Precision Timestamp Amendment

Status: **PRE-ADAPTATION STRUCTURAL AMENDMENT**

This amendment follows the frozen CoBra schema/chronology protocol and is
based only on permitted timestamp/chronology inspection. No sensor
distribution, PCA, TCN, router, target, coverage, anomaly, or energy
outcome was used.

## Observation

Three source archives use timestamps with minute rather than second
precision:

- `CoBra20240627.zip` ? TRAIN
- `CoBra20241121.zip` ? TRAIN
- `CoBra20250120.zip` ? CALIBRATION

Observed format:

`DD.MM.YYYY HH:MM`

All rows in those three archives use that timestamp form.

A timestamp-only structural probe showed:

- backward transitions: 0 for all three archives;
- minute re-entry count: 0 for all three archives;
- source rows remain in contiguous chronological minute blocks;
- typical block size is approximately 60 rows per minute;
- no seconds are present in the published timestamp values.

`CoBra20241121.zip` also contains source chronology gaps including a
76-minute transition. These gaps are preserved exactly and are not
interpolated or repaired.

## Amendment

The schema inspector may parse `%d.%m.%Y %H:%M`.

Minute-resolution timestamps are represented at their published minute
boundary only for chronology comparison. No artificial per-row second
values are generated.

The manifest records timestamp-resolution counts and explicitly records
that source row order is preserved.

Repeated parsed timestamps caused by source minute precision remain
visible through the existing duplicate-timestamp count. They are not
silently removed, deduplicated, interpolated, or expanded.

## Scientific boundary

This amendment changes only deterministic parsing of already-inspected
timestamp metadata.

It does not:

- alter the frozen 14/4/5 split;
- inspect sensor values beyond the frozen structural rules;
- use EVALUATION outcomes;
- perform model selection;
- synthesize higher-resolution timestamps;
- impute missing chronology;
- alter source-row order.

The CoBra external-validation role remains unchanged.
