# Router V2 Independent External Dataset Selection

Status: **SELECTED BEFORE ROUTER V2 DEVELOPMENT**

Selection-protocol commit:

`0278641a0b925b9d2d0c484f3bb5f445dcfffa36`

Parent closed failure-analysis commit:

`87e4d667b94ecdca0ce0a9499347e744cc48aa71`

## Selected dataset

**CoBra High Temperature Heat Pump Demonstrator — Experimental Dataset 2024–2025**

Publisher / operator:

German Aerospace Center (DLR), Cottbus, Germany

Permanent identifier:

`10.82481/cobra.2025`

Zenodo record:

`https://zenodo.org/records/15862451`

Published version considered for selection:

`1.0.0`, published 2025-09-29

Evidence role:

**FUTURE_GENUINELY_INDEPENDENT_EXTERNAL_VALIDATION**

The selected data must not be used for Router V2 model selection, feature
selection, fitting, threshold selection, or hyperparameter tuning.

## Why CoBra is selected

CoBra best satisfies the frozen preference ordering and eligibility criteria
among the reviewed candidates without inspecting Router V2 outcomes.

### Real measured compressor data

The dataset contains real sensor and actor measurements from the DLR CoBra
Brayton-cycle high-temperature heat-pump demonstrator.

The compressor is a central active component of the closed-loop Brayton-cycle
system, and the source documentation explicitly reports compressor speed,
pressure ratio, mass flow, compressor outlet temperature, and drivetrain /
vibration-related experiments.

### Strong multivariate instrumentation

Publisher metadata states that the archive contains:

- temperatures;
- pressures;
- rotational speeds;
- accelerations and derived velocities;
- humidity;
- valve positions;
- additional sensor and actor channels.

This substantially exceeds the frozen minimum of five usable numeric sensor
channels before any outcome-based filtering.

### Strong chronology

The source explicitly states:

- 1 Hz sampling;
- 23 distinct test days;
- each test-day CSV spans the whole test day from before start-up through
  shut-down;
- the experimental campaign spans June 2024 through February 2025.

This gives unambiguous chronology and enough sequential observations for causal
temporal windows and a chronological TRAIN/CAL/EVALUATION split.

### Operational diversity

The public experiment index includes steady-state runs, compressor-speed
variation, pressure/inventory variation, transient pressure-reduction tests,
emergency shutdowns, vibration/component tests, control-system tests,
recuperated/non-recuperated operation, and Carnot-battery charge/discharge
operation.

This breadth is valuable for testing whether an energy-aware compute-routing
mechanism transports across changing operating support.

### Stable public source

The dataset is published through Zenodo with DOI `10.82481/cobra.2025`.
The record lists per-test-day archive files and publisher-provided MD5 hashes,
allowing the exact source version to be locked before adaptation.

OpenAIRE identifies the dataset as CC BY.

## Independence rationale

CoBra has not been used in:

- MetroPT-3 Router V1 development;
- MetroPT2 transport studies;
- RCSD-1YD adaptation or one-shot external evaluation;
- RCSD post-hoc failure analysis;
- any Router V2 feature, model, or threshold selection.

Selection is being frozen before Router V2 development begins.

No PCA, TCN, Router V1, Router V2, anomaly, target-prevalence, coverage, or
energy metric has been computed on CoBra for this project at selection time.

## Candidate comparison

### CoBra High Temperature Heat Pump Demonstrator — SELECTED

Eligibility: **PASS**

Preference class:
`controlled_industrial_or_test_facility_telemetry`

Reasons:

- real experimental compressor system;
- compressor is central to the Brayton-cycle demonstrator;
- 1 Hz multivariate raw telemetry;
- 23 dated full test-day sequences;
- rich operating and transient diversity;
- stable DOI and per-file checksums.

### Experimental Dataset: Heat Pump coupled to a Thermal Storage for charging a Carnot Battery — ELIGIBLE, NOT SELECTED

DOI:

`10.5281/zenodo.18712882`

The University of Duisburg-Essen dataset contains raw time-series measurements
from a highly instrumented heat-pump test bench with a laboratory-scale
reciprocating compressor, including temperature, pressure, flow, and electrical
power across 17 experiments.

Non-outcome reason for not selecting:

- lower preference class (`laboratory_compressor_telemetry`);
- publisher metadata available at selection time does not state sampling
  cadence or total sequential sample count as explicitly as CoBra;
- fewer experiment days and a narrower documented acquisition campaign than
  CoBra.

### NASA High Efficiency Centrifugal Compressor Data Archive — NOT SELECTED

NASA's HECC archive contains real centrifugal-compressor experimental data from
the CE-18 compressor test facility with extensive steady-state and fast-response
instrumentation.

Non-outcome reason for not selecting:

- public archive documentation clearly establishes rich aerodynamic
  measurements and performance-map data, but the metadata reviewed for this
  selection does not establish a long, directly timestamped operational
  telemetry sequence with a simple chronological split as clearly as CoBra;
- therefore CoBra provides lower methodological ambiguity for causal temporal
  routing evaluation.

HECC remains a useful future aerodynamic / compressor-model validation source,
but it is not selected as the primary Router V2 external routing test.

### Two-Stage Metal Hydride Hydrogen Compressor Dataset — ELIGIBLE, NOT SELECTED

DOI:

`10.5281/zenodo.19036733`

The dataset contains raw measurements from 29 experiments on a two-stage metal
hydride hydrogen compressor.

Non-outcome reason for not selecting:

- the compression mechanism is thermochemical rather than the rotating
  mechanical-compressor setting used by AeroXAI's current routing studies;
- CoBra is closer to the intended physical compressor / energy-management
  setting while still being an independent domain.

### Marine Engine Fault Dataset — NOT ELIGIBLE FOR PRIMARY ROLE

DOI:

`10.5281/zenodo.19857425`

The dataset contains roughly 115,000 time-stamped samples across about 70
channels on a marine diesel-engine test bench and includes compressor
air-filter-clogging scenarios.

Non-outcome reason for rejection:

- the compressor is a subsystem within the turbocharged marine engine rather
  than the central monitored component;
- this does not satisfy the frozen primary-selection requirement as cleanly as
  CoBra.

### Compressor Performance Data for Reciprocating Piston Compressors — NOT ELIGIBLE

DOI:

`10.5281/zenodo.17599603`

Non-outcome reason for rejection:

- the 32,034-row dataset is a combined compressor-performance table across
  operating conditions and geometries;
- publisher metadata does not establish it as a chronological raw sensor
  time-series suitable for causal temporal windows.

### DATED: A Dataset of Centrifugal Compressors — NOT ELIGIBLE

DOI:

`10.5281/zenodo.8200792`

Non-outcome reason for rejection:

- it is generated from 1-D mean-line analysis simulations rather than real
  measured compressor telemetry;
- simulation-only data are explicitly ineligible under the frozen protocol.

## Data-use boundary after this selection

Allowed next:

1. download/archive the exact selected CoBra source version;
2. verify publisher checksums and compute project SHA256 hashes;
3. inspect schema, timestamp parsing, row counts, and deterministic chronology;
4. freeze the chronological TRAIN/CAL/EVALUATION split;
5. freeze a CoBra-specific adaptation and one-shot evaluation protocol.

Still forbidden until Router V2 is frozen:

- computing Router V1 or Router V2 metrics on CoBra;
- computing TCN/PCA outcome metrics on the future CoBra EVALUATION partition;
- using CoBra to choose Router V2 features, model family, hyperparameters, or
  thresholds;
- changing the selected dataset because it appears difficult.

## Selection decision

Lock:

`CoBra High Temperature Heat Pump Demonstrator — Experimental Dataset 2024–2025`

as the third independent compressor-domain resource for the future Router V2
one-shot external validation.

The exact source files, checksums, schema, and chronological split will be
frozen in the next source-lock step before Router V2 development.
