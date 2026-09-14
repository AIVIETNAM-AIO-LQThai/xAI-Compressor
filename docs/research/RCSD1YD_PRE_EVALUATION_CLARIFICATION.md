# RCSD-1YD Pre-EVALUATION Clarification

## Timing

This clarification is recorded after RCSD-1YD TRAIN/CALIBRATION adaptation and
before any EVALUATION sensor summary, TCN metric, router metric, or energy
metric has been computed.

It does not change the frozen model, router, thresholds, feature definitions,
energy protocol, or numeric success criteria.

## Reason

The preregistered primary metrics define coverage of EVALUATION TCN evidence
using CALIBRATION-derived q0.90 and q0.995 thresholds, but the original protocol
did not explicitly define the zero-denominator case.

A coverage fraction is undefined when EVALUATION contains zero teacher-positive
bins. This must be resolved before EVALUATION is opened.

## Frozen rule

High-evidence coverage:

`routed high-evidence bins / all EVALUATION high-evidence bins`

Alert-evidence coverage:

`routed alert-evidence bins / all EVALUATION alert-evidence bins`

If EVALUATION contains **zero high-evidence bins**, the external evaluation is
classified:

`INCONCLUSIVE_EXTERNAL_EVALUATION`

and cannot be declared PASS.

If EVALUATION contains **zero alert-evidence bins**, the external evaluation is
classified:

`INCONCLUSIVE_EXTERNAL_EVALUATION`

and cannot be declared PASS.

No minimum count larger than one is introduced. If the denominator is positive,
the preregistered >=80% coverage criterion applies exactly as written.

There is no rescue score and no post-EVALUATION threshold change.

## Scientific effect

This clarification cannot make a failing router pass. It only prevents an
undefined 0/0 coverage case from being interpreted as success.
