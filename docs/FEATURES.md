# MetroPT Feature Contract

## Temporal resolution

Raw MetroPT-3 telemetry is aggregated into trailing
5-minute bins.

Bins are right-labelled and right-closed.

No interpolation, backward filling, centered rolling
window, or future-looking transformation is permitted.

## Data quality

Expected observations per bin: 30

Minimum accepted coverage: 80%

Bins below the coverage threshold are excluded.

Coverage information is retained for quality control but
is never supplied to anomaly models.

## Analog features

For each analog sensor:

- mean
- standard deviation
- minimum
- maximum
- last observed value

## Digital features

For each digital sensor:

- active ratio
- transition count
- last observed state

## Derived pressure relationships

TP3 - Reservoirs

TP2 - TP3

For each:

- mean
- standard deviation

## Model dimensionality

63 model features.

Quality metadata:

- sample_count
- coverage_ratio

Quality metadata must never be included in anomaly-model
inputs.

## Data partitions

Training:
2020-02-01 through 2020-02-21

Calibration:
2020-02-22 01:00 through 2020-02-29

Holdout test:
2020-03-01 01:00 through the end of the dataset

Documented failure events remain outside training and
calibration.

## Sequence contract

Future temporal models may use 12 consecutive bins,
corresponding to 60 minutes of historical context.

No sequence may cross a train/calibration/test boundary.
