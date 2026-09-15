# AeroXAI Cross-Dataset Research Synthesis

Status: **research synthesis**

This document consolidates the detector, explanation, calibration,
energy-aware routing, and simulation evidence accumulated after the frozen V3
release.

It does not promote an exploratory detector or router into production and does
not convert research evidence into a causal physical diagnosis.

## Research questions

The program developed along two connected questions.

Detector/explanation question:

> Is concentrated anomaly evidence an artifact of MetroPT-3, an intrinsic
> property of PCA reconstruction, or an interaction among data, representation,
> scaling, sensor availability, state, and time?

Energy-aware intelligence question:

> Can AeroXAI preserve high-value temporal evidence while avoiding unnecessary
> expensive AI inference, and do logical inference savings translate into
> measured compute-energy savings on new data?

The evidence supports a qualified answer to both:

1. explanation concentration is pipeline- and data-dependent rather than an
   intrinsic property of PCA; and
2. selective temporal inference can preserve important evidence, but compute
   savings do not automatically transport in magnitude across domains.

## Evidence hierarchy

The evidence types must remain separated.

### REAL telemetry / recordings

- MetroPT-3 historical APU process telemetry.
- MetroPT2 historical APU process telemetry.
- IITK Air Compressor acoustic recordings with seeded health-state faults.
- RCSD-1YD centrifugal-compressor telemetry used for external router stress
  testing.
- CoBra high-temperature heat-pump demonstrator telemetry used for
  preregistered within-domain adaptation followed by future chronological EVAL.

### MEASURED AI-compute energy

- GPU-device energy for TCN scoring measured with NVML.
- These measurements do not include CPU/router energy or whole-system power
  unless explicitly stated.

### SIMULATED physics/control

- AeroXAI inverse-state physics.
- bounded physical uncertainty.
- robust compressor dispatch.
- proof-carrying 60-second advisory action.

The REAL detector/routing studies do not calibrate the synthetic compressor
twin. The SIMULATED control chain therefore remains a separate validation path.

---

## 1. Frozen MetroPT-3 baseline

The frozen Robust-PCA detector uses:

```text
RobustScaler
PCA reconstruction error
5-minute causal feature bins
calibration-only thresholding
causal EWMA
3-of-4 persistence
```

The frozen detector detected all four documented high-stress air-leak
incidents within the predefined timely window:

```text
4 / 4 timely
2 / 4 pre-onset
2 / 4 shortly after onset
false alert rate ≈ 0.886 / day
PR-AUC ≈ 0.249
```

Approved wording:

> PCA detected all four documented incidents within the predefined timely
> window; two produced pre-onset warnings and two were detected shortly after
> onset.

Do not claim all four incidents were predicted in advance.

### Frozen explanation concentration

At the four incident explanation timestamps:

```text
mean top-3 contribution share ≈ 0.99996
mean effective group count N_eff ≈ 1.72
range N_eff ≈ 1.26 to 2.10
```

This describes detector evidence, not physical fault complexity.

---

## 2. MetroPT2 cross-period / cross-fault replication

MetroPT2 tested whether the concentrated Robust-PCA evidence was specific to
MetroPT-3.

The frozen preprocessing protocol used:

```text
full
flowmeter_excluded
strict_flow_blind
```

and evaluated an air-leak event plus an oil-leak event.

### Main result

Robust-PCA concentration persisted across:

- a different MetroPT recording period;
- two fault labels;
- inclusion/exclusion of Flowmeter;
- stricter flow-blind variants.

Therefore MetroPT-3 alone did not explain the concentration.

Approved conclusion:

> Multi-dataset evidence did not support the hypothesis that MetroPT-3 alone
> caused the highly concentrated explanations.

---

## 3. Representation and scaling ablation

A frozen 2×2 study compared:

```text
RobustScaler   + PCA reconstruction
StandardScaler + PCA reconstruction
RobustScaler   + direct feature energy
StandardScaler + direct feature energy
```

The matched-time comparison showed that concentration depended on:

```text
data geometry
scaling
representation
sensor availability
fault/state
explanation timestamp
```

Standard-PCA was often less concentrated than Robust-PCA at matched times, and
full-feature Standard methods could become dominated by Flowmeter.

Approved conclusion:

> Explanation concentration was not invariant to the analysis pipeline.

---

## 4. Temporal explanation dynamics

The next experiment traced explanation concentration over fixed
incident-relative windows.

A broad-evidence state sometimes preceded a persistent event, but high
effective dimensionality also occurred in non-alerting periods.

Therefore:

> High-dimensional evidence alone is not a precursor.

The relevant object is the joint behavior of anomaly magnitude and explanation
spread.

---

## 5. Calibration-normalized evidence states

A calibration-relative state experiment tested a candidate sequence:

```text
subthreshold|diffuse
→ threshold_crossing|diffuse
→ alerting|concentrated
```

The universal sequence failed.

Different representations entered the alerting regime with different
concentration states.

Scientific value:

> The negative result prevented fitting a universal state-transition narrative
> to a very small repeatedly inspected incident set.

---

## 6. Independent IITK compressor/acoustic validation

The IITK Air Compressor Health State Dataset provided an independent
cross-asset and cross-modality test.

Frozen split:

```text
Healthy Reading001-120      train
Healthy Reading121-180      calibration
Healthy Reading181-225      test
All 1,575 fault recordings  test only
```

Detection:

| Representation | ROC-AUC | Healthy FPR | Fault recall | Balanced accuracy |
|---|---:|---:|---:|---:|
| Robust-PCA | 1.0000 | 0.0% | 99.746% | 99.873% |
| Standard-PCA | 1.0000 | 0.0% | 99.937% | 99.968% |
| Standard feature-energy | 0.99898 | 0.0% | 94.857% | 97.429% |
| Robust feature-energy | 0.97027 | 2.22% | 38.095% | 67.937% |

Unlike MetroPT, IITK PCA explanations remained substantially distributed
across acoustic feature families.

Therefore IITK directly rejected:

> PCA reconstruction intrinsically produces concentrated explanations.

Calibration-resampling analysis also showed substantially stronger operating
stability for the PCA variants than for direct feature-energy.

---

## 7. Detector-side conclusion before energy-aware routing

By the end of the detector program, the most defensible statement was:

> A deployable anomaly representation should not be selected by ranking
> performance alone. It should also provide inspectable evidence and a stable
> calibration-to-action boundary.

PCA remained the most defensible anomaly representation in the AeroXAI
research evidence because it combined:

- strong frozen MetroPT incident detection;
- decomposable reconstruction evidence;
- cross-asset IITK ranking performance;
- stronger IITK calibration stability than direct feature-energy.

This did not promote a new detector into the frozen V3 product.

---

## 8. Energy-aware hierarchical intelligence

The energy-aware program extended the research question from:

```text
What evidence should we compute?
```

to:

```text
When is expensive evidence worth computing?
```

The hierarchy separates cheap evidence from more expensive temporal inference.

Conceptually:

```text
cheap detector / support evidence
→ selective temporal model
→ physical verification gate
→ optimizer / robust action
→ explanation
```

The temporal model remains evidence generation, not autonomous control.

The energy objective includes both plant-side and AI-compute-side cost, but the
available evidence must keep them separate:

```text
plant energy      SIMULATED in the current architecture studies
GPU TCN energy    MEASURED in the routing benchmarks
CPU energy        generally UNKNOWN
whole-system AI   generally UNKNOWN
```

---

## 9. Evidence-aware Router V1 and RCSD external stress test

The first evidence-aware router used cheap PCA-derived and support-derived
signals to decide when to invoke the TCN.

Development results suggested meaningful compute savings on MetroPT while
retaining a large fraction of teacher-defined high-value evidence.

A genuinely new RCSD-1YD centrifugal-compressor evaluation then exposed a
transport failure.

Frozen RCSD result:

```text
TCN invocation fraction        ≈ 37.07%
high-evidence coverage         ≈ 39.12%
alert-evidence coverage        ≈ 50%
mean gross GPU-energy reduction ≈ 57.00%
energy reduction positive       5 / 5 repetitions
```

The joint external criteria failed because evidence retention did not
transport.

This was scientifically important:

> Compute reduction transported, but the evidence-selection rule did not.

Post-hoc diagnosis showed that the cheap PCA representation itself had not
simply collapsed. Instead, several support-derived inputs changed behavior
under the new domain, while the frozen logistic combination lost useful
transport.

RCSD EVAL was then reclassified as development data for Router V2. It was not
reused as independent validation for V2.

---

## 10. Router V2 redesign

Router V2 deliberately simplified the online input space to exactly three
PCA-derived variables:

```text
pca_ewma_percentile
recent_q90_hit_fraction
recent_q95_hit_fraction
```

No TCN score and no incident label is used online.

Frozen family:

```text
max3
```

Frozen threshold:

```text
0.40
```

The V2 development criterion prioritized evidence coverage before compute
reduction.

Development domains:

```text
MetroPT-3
MetroPT2
RCSD-1YD
```

Because RCSD was already seen, a new external dataset was required for the
primary Router V2 claim.

---

## 11. CoBra adaptation protocol

CoBra was selected as the new domain before its sensor-value outcomes were
used for Router V2 evaluation.

The study was not zero-shot. It used preregistered within-domain adaptation:

```text
TRAIN  first 14 days
CAL    next 4 days
EVAL   final 5 days
```

The final five EVAL days were held until:

- the 219-feature raw contract was frozen;
- the 5-minute representation was frozen;
- PCA and TCN architecture/training were frozen;
- CAL-only thresholds were frozen;
- Router V2 was frozen;
- the one-shot EVAL implementation was committed.

No EVAL-specific feature dropping, refit, threshold adjustment, or router
redesign was allowed.

---

## 12. CoBra one-shot future evaluation

Frozen EVAL:

```text
source rows                125,826
valid 5-minute bins            358
eligible TCN targets           247
high-evidence units             93
alert-evidence units             1
```

Router V2:

```text
TCN invocations                194 / 247
TCN invocation fraction         78.54%
TCN call reduction              21.46%
high-evidence coverage         100.00%
alert-evidence coverage        100.00% descriptive only
```

Preregistered evidence verdict:

```text
high evidence   PASS
alert evidence  INSUFFICIENT_EVIDENCE
```

Only one alert-evidence unit occurred, below the required minimum of three.

The supported evidence-retention conclusion is therefore:

> After preregistered within-domain adaptation, Router V2 retained all 93
> adequate high-evidence TCN units on the future CoBra EVAL partition.

Do not convert this into a zero-shot or universal-routing claim.

---

## 13. CoBra measured GPU-energy result

The energy benchmark compared:

```text
always-on TCN: 247 invocations / logical pass
Router V2:     194 invocations / logical pass
```

Measurement scope:

```text
NVIDIA GeForce RTX 4090
GPU-device TCN scoring only
NVML sampling
5 measured repetitions
matched idle
minimum 20-second workloads
```

Gross GPU-energy reduction by repetition:

```text
9.89%
8.39%
6.58%
11.98%
9.92%
```

Mean:

```text
9.35%
```

The routed condition used less gross GPU energy in all five repetitions.

However, the preregistered criterion required:

```text
mean gross reduction >= 50%
and positive reduction in all five repetitions
```

Therefore:

```text
GPU-energy criterion   FAIL
overall CoBra status   FAIL
```

The correct interpretation is:

> Evidence selection transported better than compute efficiency. Router V2
> preserved high-evidence coverage on CoBra, but the logical 21.46% TCN-call
> reduction translated to only 9.35% measured gross GPU-energy reduction.

This is a negative result against the frozen energy target and must remain so.

---

## 14. Cross-dataset energy-aware conclusion

The Router V1 → RCSD → Router V2 → CoBra sequence shows two different transport
failures.

### RCSD

```text
compute reduction transported
evidence selection did not
```

### CoBra Router V2

```text
high-evidence selection transported
required compute-energy reduction did not
```

Together these results argue against treating routing as a single scalar
optimization problem.

A useful deployment objective is multi-criteria:

```text
evidence retention
+
compute invocation
+
measured compute energy
+
latency
+
support validity
+
physical risk
```

Logical call reduction is not a sufficient proxy for measured GPU energy.

---

## 15. Integrated research conclusions

### Supported

1. **PCA reconstruction does not intrinsically cause sparse explanations.**

2. **Explanation concentration is pipeline-, data-, state-, and time-dependent.**

3. **Ranking quality alone is insufficient for deployable anomaly detection.**

4. **Calibration stability matters.**

5. **Selective temporal inference can preserve high-value evidence on new
   chronological data after preregistered within-domain adaptation.**

6. **Logical inference reduction and measured GPU-energy reduction are distinct
   quantities.**

7. **Negative external results are essential for routing design.**

   RCSD prevented promotion of Router V1; CoBra prevents overclaiming Router
   V2's energy efficiency.

8. **AeroXAI's strongest current architecture is hierarchical and gated rather
   than monolithic.**

### Not supported

Do not claim:

- PCA predicts every compressor fault in advance.
- concentrated detector evidence means a physically simple fault.
- distributed evidence alone is a precursor.
- AeroXAI learned a universal fault-progression state machine.
- sampled IITK AUC=1 means population AUC=1.
- Router V1 or Router V2 is universally transportable.
- CoBra validates zero-shot routing.
- CoBra proves alert-level routing transport.
- a 21.46% call reduction equals a 21.46% GPU-energy reduction.
- the measured 9.35% GPU reduction is whole-system AI energy.
- the REAL datasets calibrate the current physics twin.
- the REAL studies validate simulated plant-energy savings.
- AeroXAI autonomously controls compressor equipment.

---

## 16. Research principle for AeroXAI

The detector-side principle remains:

> **AeroXAI should not judge an anomaly representation by ranking performance
> alone. A deployable representation must also produce interpretable evidence
> and support a statistically stable calibration-to-action boundary.**

The energy-aware extension is:

> **AeroXAI should not judge an inference router by call reduction alone. A
> deployable router must preserve valuable evidence under support shift and
> demonstrate measured compute benefit under the actual execution stack.**

Together these principles support the proof-carrying architecture:

```text
telemetry
→ cheap anomaly evidence
→ selective temporal evidence
→ explanation
→ hypothesis
→ physical verification
→ robust action
→ proof bundle
```

The architecture remains advisory/shadow unless the physical bridge has been
independently calibrated and validated.
