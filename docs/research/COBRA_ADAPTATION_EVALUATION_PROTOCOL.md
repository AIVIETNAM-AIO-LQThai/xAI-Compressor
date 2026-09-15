# CoBra Adaptation and External Evaluation Protocol

Status: **FREEZE BEFORE ROUTER V2 DEVELOPMENT**

Parent schema/chronology result:

`3b6b2f358c80fe4aafe2506c493a8bcb680b1fb6`

Source lock:

`9d3e535ae7e6932fce1a4ec975d0ebacdfa4b0f6`

Evidence role:

`FUTURE_GENUINELY_INDEPENDENT_EXTERNAL_VALIDATION`

## Purpose

Freeze the CoBra dataset-side adaptation and future-evaluation contract before
Router V2 architecture, mathematical form, coefficients, thresholds, temporal
representation, or hyperparameters are selected.

This protocol does not define Router V2.

Router V2 must be developed using already-consumed development evidence.
CoBra model outcomes may not influence Router V2 architecture or
hyperparameter selection.

CoBra is not intended to be a pure zero-shot external-domain test.

The intended claim is:

**externally selected domain with preregistered within-domain TRAIN/CAL
adaptation and one-shot future EVALUATION.**

## Frozen source and split

The committed CoBra source remains frozen.

The whole-test-day split remains unchanged:

- TRAIN: first 14 frozen test days;
- CALIBRATION: next 4 frozen test days;
- EVALUATION: final 5 frozen test days.

No day may be moved, removed, divided, or reassigned because of:

- row count;
- timestamp resolution;
- schema differences;
- missing channels;
- operating behavior;
- model scores;
- evidence prevalence;
- routing performance;
- energy results;
- later EVALUATION results.

## Exact-name schema alignment

All future CoBra ingestion must align columns by exact column name.

Positional feature alignment is prohibited.

Column order differences between source archives must never alter feature
identity.

## Structural compatibility envelope

The schema/chronology result established that not every source archive exposes
the same column set.

Therefore a structural compatibility envelope must be derived mechanically
from committed schema metadata.

The candidate envelope is defined as:

1. intersect exact column names across all 23 frozen archives;
2. exclude the timestamp column;
3. retain only fields whose committed coarse schema class is `numeric` in
   every archive;
4. order surviving names lexicographically by exact source name;
5. serialize the ordered candidate list and SHA256 before any CoBra
   model outcome is opened.

This candidate envelope is not automatically the final model feature set.

## Final raw-feature contract

A separate metadata-only feature contract may be frozen before CoBra model
outcomes are opened.

That contract may use only:

- committed exact column names;
- committed coarse schema classes;
- committed source metadata such as `overview_experiments.csv`;
- externally documented sensor semantics;
- requirements already imposed by the frozen Router V2 policy.

It may not use:

- CoBra sensor values;
- CoBra sensor distributions;
- TRAIN/CAL/EVALUATION model scores;
- reconstruction errors;
- TCN losses;
- routing performance;
- evidence prevalence;
- energy results;
- EVALUATION behavior.

Feature inclusion or exclusion may not be justified by observed CoBra model
performance.

Any final feature list must be exact-name ordered and hashed before model
outcomes are opened.

## Missing-feature rule

A model-required feature must either:

1. belong to the frozen universal compatible feature contract; or
2. have a deterministic missing-feature behavior frozen before CoBra model
   outcomes are opened.

EVALUATION-specific imputation, substitution, feature dropping, or rescue
logic is prohibited.

A required feature becoming unavailable without a preregistered behavior is a
processing failure for the affected sample/day.

No rule may be invented after inspecting EVALUATION model behavior.

## Timestamp semantics

Published timestamps are treated as source chronology metadata.

The actual physical timezone is not inferred from the source files.

Any timezone attached internally by parsing code is representational only and
must not be interpreted as evidence that the original source timestamps are
UTC.

No cross-day elapsed-time inference is allowed.

Three frozen source days publish only minute-resolution timestamps.

For those days:

- original source-row order must be preserved;
- no artificial seconds may be generated;
- repeated minute timestamps must remain visible;
- within-minute physical elapsed time must not be inferred;
- source row order may be used for deterministic aggregation but not as proof
  of one-second physical timing.

## Temporal representation

This protocol does not freeze a specific temporal bin width, history length,
prediction horizon, or temporal-model architecture.

Those are Router V2 / temporal-model design choices and must be frozen before
CoBra TRAIN/CAL model outcomes are computed.

Any later temporal representation must satisfy all of the following:

- deterministic;
- causal;
- identical for TRAIN, CALIBRATION, and EVALUATION;
- compatible with the source timestamp resolution;
- no fabricated timestamps;
- no interpolation across genuine source gaps;
- no forward-fill or backward-fill across source gaps unless such behavior was
  independently preregistered before CoBra model outcomes;
- no temporal window may cross a frozen test-day boundary;
- temporal state resets at day boundaries;
- temporal state resets when the frozen representation identifies an
  unbridgeable source chronology gap.

If a later frozen Router V2 policy requires temporal precision unavailable in a
CoBra source day, the response to that incompatibility must be specified before
CoBra model outcomes are opened.

## TRAIN adaptation

CoBra TRAIN may be used for learned dataset-specific adaptation only after the
Router V2 policy and the CoBra feature/temporal contract are frozen.

TRAIN may then fit only parameters explicitly permitted by those frozen
policies, for example:

- feature scaling parameters;
- PCA or other low-level evidence models;
- temporal-model weights;
- other dataset-specific parameters explicitly required by the frozen
  hierarchy.

CoBra TRAIN may not be used to redesign:

- Router V2 architecture;
- router mathematical form;
- router feature families;
- temporal-model architecture;
- history length;
- prediction horizon;
- router hyperparameters;
- success criteria.

If the frozen model performs poorly on TRAIN, that does not authorize redesign
using CoBra.

## CALIBRATION

CALIBRATION may be used only for operations explicitly declared before its
model outcomes are inspected.

Examples may include:

- empirical reference distributions;
- fixed percentiles;
- rank transforms;
- thresholds;
- normalization constants;
- other calibration quantities explicitly required by the frozen Router V2
  policy.

CALIBRATION may not be used to redesign the router or temporal model.

No EVALUATION information may be used while deriving CALIBRATION quantities.

All final CoBra adaptation artifacts must be frozen before EVALUATION is
opened.

## Router V2 evidence definitions

This protocol does not independently redefine `high evidence` or
`alert evidence`.

Their exact definitions must be inherited from the later frozen Router V2
policy.

Those definitions must be frozen before CoBra model outcomes are opened and
must be applied to CoBra without EVALUATION-specific reinterpretation.

## Primary evidence-retention criteria

The future external EVALUATION criteria are:

- high-evidence coverage >= 80%, provided at least 20 eligible high-evidence
  EVALUATION units exist;
- alert-evidence coverage >= 80%, provided at least 3 eligible alert-evidence
  EVALUATION units exist.

The definition of one eligible evaluation unit follows the later frozen Router
V2 temporal/evidence policy.

If the corresponding minimum count is not reached, report:

`INSUFFICIENT_EVALUATION_SUPPORT`

The criterion must not be silently removed, redefined, or replaced.

No incident-label criterion may be introduced unless independently frozen
ground-truth labels are established before EVALUATION.

## Compute-energy criterion

The primary compute-energy criterion is:

- mean gross GPU TCN-scoring energy reduction >= 50% relative to always-on TCN;
- gross GPU TCN-scoring energy reduction must be positive in all 5
  preregistered repetitions.

The primary energy claim is restricted to directly measured GPU TCN scoring
unless a separate preregistered measurement expands that scope.

Do not infer:

- CPU/router energy;
- total host energy;
- whole-system energy;
- compressor plant energy;
- carbon savings;

from GPU TCN measurements alone.

Energy measurement should use the same frozen measurement methodology as the
earlier hierarchical-compute experiments when compatible GPU/NVML measurement
hardware is available.

If compatible GPU/NVML measurement hardware is unavailable, report the
energy criterion as:

`NOT_EVALUATED`

Do not substitute CPU execution time, estimated TDP, nominal GPU power, or
other unmeasured proxies for the required GPU energy result.

A CPU-only machine may still be used for:

- protocol work;
- metadata checks;
- ingestion tests;
- deterministic correctness tests;
- non-energy research implementation.

## EVALUATION firewall

The five frozen EVALUATION days form the one-shot future evaluation period.

Before the final EVALUATION run, freeze and hash all applicable:

- feature contracts;
- temporal representation;
- preprocessing rules;
- scalers;
- PCA/evidence models;
- temporal-model weights;
- calibration reference distributions;
- thresholds;
- router policy;
- execution config;
- evaluation script.

During or after EVALUATION, the following are prohibited:

- model fitting;
- scaler refitting;
- PCA refitting;
- temporal-model training;
- threshold selection;
- feature selection;
- temporal-window redesign;
- architecture selection;
- router coefficient fitting;
- fallback tuning;
- EVALUATION-specific imputation;
- EVALUATION-specific feature substitution;
- rescue experiments presented as the preregistered result.

If the frozen procedure cannot process an EVALUATION case as specified, report
the preregistered failure/unevaluable outcome.

Do not change the procedure and rerun the external test as though it remained
independent.

## Independence firewall before Router V2 freeze

Before Router V2 is frozen, CoBra may contribute only:

- committed source provenance;
- frozen split metadata;
- frozen schema metadata;
- frozen chronology metadata;
- source metadata permitted by the schema protocol;
- this adaptation/evaluation contract;
- metadata-only feature-contract derivation.

Before Router V2 is frozen, do not compute on CoBra:

- sensor descriptive statistics;
- sensor quantiles;
- sensor correlations;
- PCA scores;
- reconstruction error;
- EWMA evidence;
- support scores;
- TCN prediction loss;
- Router V1 probabilities;
- Router V2 probabilities;
- teacher/evidence prevalence;
- routing coverage;
- anomaly rate;
- energy outcomes;
- TRAIN/CAL/EVALUATION sensor-value comparisons.

## Required development sequence

After this protocol is frozen:

1. derive and freeze only metadata-level CoBra compatibility artifacts;
2. close the CoBra protocol-design stage;
3. develop Router V2 using already-consumed development evidence only;
4. freeze Router V2 architecture, mathematical rule/model, feature interface,
   temporal requirements, hyperparameters, and evidence definitions;
5. return to CoBra;
6. freeze the final CoBra feature and temporal implementation contracts before
   opening model outcomes;
7. execute allowed TRAIN adaptation;
8. execute allowed CALIBRATION;
9. freeze every resulting adaptation artifact;
10. open EVALUATION exactly once.

If frozen Router V2 requirements prove incompatible with CoBra, document the
incompatibility.

Do not redesign Router V2 using CoBra outcomes.

## Claim boundary

A successful result may support:

**transport of the frozen hierarchical routing approach to an externally
selected controlled industrial heat-pump/compressor domain after
preregistered within-domain TRAIN/CAL adaptation and one-shot future
EVALUATION.**

It does not establish:

- universal compressor generalization;
- zero-shot cross-domain generalization;
- causal fault diagnosis;
- autonomous industrial control;
- measured plant-energy savings;
- measured carbon savings.

All later claims must preserve the existing AeroXAI provenance and advisory
boundaries.
