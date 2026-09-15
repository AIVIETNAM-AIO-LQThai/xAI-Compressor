# CoBra TRAIN Numeric Parsing Compatibility Amendment

## Timing

This amendment was made after opening CoBra TRAIN sensor values but before
fitting or scoring PCA, EWMA, TCN, Router V2, teacher thresholds, or energy
metrics.

CALIBRATION and EVALUATION sensor values had not been opened.

## Observed incompatibility

The first frozen TRAIN ingestion produced:

- 310,869 source rows;
- 1,086 produced five-minute bins;
- 0 valid five-minute bins.

The existing numeric parser used ordinary decimal-point coercion.

CoBra source sensor values use European decimal-comma numeric notation in the
semicolon-delimited source CSV representation. Consequently, valid physical
sensor strings were interpreted as nonnumeric.

This was a source-format compatibility failure, not a model-performance result.

## Correction

Strict numeric coercion now accepts both:

- ordinary decimal-point notation, for example `12.5`;
- decimal-comma notation, for example `12,5`.

A decimal comma is converted to a decimal point only when the entire stripped
string matches a single numeric decimal-comma grammar, including optional sign
and exponent.

Strings containing ambiguous mixed separators such as `1.234,56` remain
invalid rather than being guessed.

All other malformed, blank, NaN, or infinite handling remains unchanged.

## Frozen boundaries unchanged

This correction does not change:

- the 219-feature contract;
- feature ordering or SHA identity;
- TRAIN/CALIBRATION/EVALUATION split;
- five-minute aggregation;
- invalid-bin rule;
- interpolation policy;
- gap handling;
- PCA specification;
- TCN architecture or hyperparameters;
- Router V2 policy `max3 >= 0.40`;
- CALIBRATION thresholds;
- EVALUATION protocol.

No feature was dropped because of TRAIN behavior.

No CALIBRATION or EVALUATION information was used.

The amendment is therefore treated as a transparent source-format
compatibility correction, not rescue tuning.
