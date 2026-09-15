# Evidence-Aware Router V2 Protocol

Status: **PREREGISTRATION ? NO ROUTER V2 RESULT EXECUTED**

Branch:

`research/evidence-aware-router-v2`

Frozen parent:

`1000a7e096136c32ecc59844a7ca67966baec8e1`

CoBra structural compatibility contract:

`1000a7e096136c32ecc59844a7ca67966baec8e1`

## Purpose

Develop a transport-robust cheap escalation policy after the closed failure
of the pooled logistic Router V1 on RCSD-1YD.

Router V2 is developed only from already-consumed compressor evidence.

CoBra model outcomes remain unopened and may not influence Router V2 design,
candidate selection, thresholds, or interpretation.

## Scientific motivation

The consumed RCSD failure analysis showed that Router V1 failed primarily
because its learned cross-domain feature combination did not transport.

The failure was not a wholesale loss of cheap PCA evidence.

The strongest transporting cheap signals included:

- PCA EWMA percentile;
- recent q90 hit fraction;
- recent q95 hit fraction.

Support-related inputs degraded strongly and are therefore excluded from the
primary Router V2 candidate family.

Router V2 tests whether a simpler monotone combination of transporting,
domain-normalized signals is more robust than a pooled learned logistic map.

## Development datasets

Allowed development evidence:

- consumed MetroPT-3 router-development evidence;
- consumed MetroPT2 router-development evidence;
- consumed RCSD-1YD EVALUATION evidence and its frozen failure analysis.

These datasets no longer have independent-validation status for Router V2.

Forbidden development evidence:

- CoBra TRAIN sensor/model outcomes;
- CoBra CALIBRATION sensor/model outcomes;
- CoBra EVALUATION sensor/model outcomes.

The already-frozen CoBra structural and adaptation protocol metadata may be
read only to enforce the independence boundary.

## Online Router V2 inputs

Router V2 uses exactly three cheap causal inputs:

1. `pca_ewma_percentile`
2. `recent_q90_hit_fraction`
3. `recent_q95_hit_fraction`

Denote them:

- x1 = `pca_ewma_percentile`
- x2 = `recent_q90_hit_fraction`
- x3 = `recent_q95_hit_fraction`

All three are domain-normalized quantities in [0, 1].

TCN score is never an online router input.

Incident labels are never router inputs.

Support-distance variables from Router V1 are not part of the primary V2
candidate family.

## Candidate family A ? max3

Score:

`max(x1, x2, x3)`

## Candidate family B ? top2_mean

Sort x1, x2, x3 and average the two largest values.

## Candidate family C ? monotone convex score

The only allowed weight tuples are:

- [1.00, 0.00, 0.00]
- [0.75, 0.25, 0.00]
- [0.75, 0.00, 0.25]
- [0.50, 0.50, 0.00]
- [0.50, 0.25, 0.25]
- [0.50, 0.00, 0.50]

Score:

`w1*x1 + w2*x2 + w3*x3`

All weights are nonnegative and sum to one.

Therefore increasing any individual Router V2 input cannot decrease the score.

No coefficients are learned from pooled cross-domain data.

## Routing thresholds

Frozen candidate thresholds:

- 0.25
- 0.30
- 0.35
- 0.40
- 0.45
- 0.50
- 0.55
- 0.60
- 0.65
- 0.70
- 0.75
- 0.80
- 0.85
- 0.90
- 0.95

The candidate routes when:

`score >= threshold`

Total candidate count:

- max3: 15
- top2_mean: 15
- convex: 90
- total: 120

No candidate family, weight, or threshold may be added after development
results are observed.

## Missing-input behavior

If the TCN target is valid but one or more Router V2 inputs cannot be
constructed causally, Router V2 must escalate to the TCN.

No imputation is used to avoid escalation.

If the TCN target itself is temporally invalid, the row is not an eligible
routing unit.

## Offline teacher definitions

The teacher is temporal-model evidence, not physical fault truth.

For each consumed development domain, retain the already-frozen teacher
threshold semantics from the source experiment.

High temporal evidence:

`TCN score >= frozen CAL-derived q0.90 threshold`

Alert temporal evidence:

`TCN score >= frozen CAL-derived q0.995 threshold`

Do not recompute a new teacher threshold from development EVALUATION rows.

Do not redefine the RCSD teacher because its temporal-evidence prevalence
shifted.

## Development adequacy

A domain contributes to high-evidence eligibility only if it contains at
least 20 eligible high-evidence units.

An alert constraint applies to a domain only if it contains at least 3
eligible alert-evidence units.

Otherwise alert evaluation for that domain is reported as
`INSUFFICIENT_ALERT_SUPPORT`.

## Candidate eligibility

For every development domain with adequate high-evidence support:

`high_evidence_coverage >= 0.80`

For every development domain with adequate alert support:

`alert_evidence_coverage >= 0.80`

No weighted average can compensate for failure on one adequately supported
domain.

## Candidate selection

Among eligible candidates:

1. minimize worst-domain TCN invocation fraction;
2. minimize mean-domain TCN invocation fraction;
3. maximize worst-domain high-evidence coverage;
4. maximize worst-domain applicable alert-evidence coverage;
5. prefer candidate family in this order:
   - max3
   - top2_mean
   - convex
6. prefer the higher routing threshold;
7. for convex-family ties, prefer the earlier weight tuple in the frozen list.

If no candidate is eligible, close this Router V2 family.

Do not rescue-tune thresholds, add features, or alter weights.

## CoBra firewall

Before Router V2 policy freeze, do not compute from CoBra:

- raw sensor statistics;
- PCA scores;
- reconstruction errors;
- EWMA scores;
- TCN scores;
- Router V2 scores;
- teacher prevalence;
- routing coverage;
- energy;
- TRAIN/CAL/EVALUATION comparisons.

The CoBra structural compatibility envelope is not a Router V2 development
dataset.

## Future CoBra adaptation

Only after Router V2 selection and policy freeze may a separate CoBra
adaptation stage define, before CoBra model outcomes:

- final raw sensor mapping;
- numeric parsing contract;
- temporal representation;
- PCA training semantics;
- TCN training semantics;
- CAL-only percentile references;
- q90/q95 hit definitions;
- gap/reset behavior.

Router V2 mathematical form, candidate-selected weights, and routing threshold
may not change because of CoBra performance.

## Future external success criteria

The already-frozen CoBra protocol remains authoritative:

- high-TCN-evidence coverage >= 80% when adequately supported;
- alert-evidence coverage >= 80% when adequately supported;
- mean gross GPU TCN-scoring energy reduction >= 50%;
- positive gross GPU TCN-scoring energy reduction in all 5 repetitions.

## Claim boundaries

Router V2 development is not independent validation.

Consumed MetroPT and RCSD results are development evidence.

TCN evidence is not physical fault truth.

No causal diagnosis claim follows from Router V2.

No autonomous control claim follows from Router V2.

No plant-energy or carbon-saving claim follows from GPU routing results.

Only the later frozen CoBra EVALUATION can provide the new external test.
