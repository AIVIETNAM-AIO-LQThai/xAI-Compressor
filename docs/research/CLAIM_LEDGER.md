# AeroXAI Claim Ledger

Status: **research synthesis**

Purpose: define exactly what can and cannot be claimed after the frozen V3
release plus the multi-dataset detector and energy-aware routing research
program.

| Claim | Evidence | Status | Allowed wording | Do not say |
|---|---|---|---|---|
| Frozen PCA detected all four documented MetroPT-3 incidents in the timely window | REAL MetroPT-3 | Supported | "Detected all four within the predefined timely window; two pre-onset, two shortly after onset." | "Predicted all four in advance." |
| Frozen PCA explanations identify root cause | MetroPT-3 XAI | Not supported | "Signals contributing to detector evidence." | "Confirmed root cause." |
| Counterfactual repair verifies detector dependence | MetroPT-3 XAI | Supported | "Removing highlighted anomalous model-space evidence caused the frozen alert to disappear." | "Repairing that sensor would fix the machine." |
| MetroPT-3 alone causes sparse explanations | MetroPT2 replication | Rejected | "Concentration persisted on MetroPT2." | "MetroPT-3 is too simple and therefore explains only one signal." |
| PCA alone causes sparse explanations | MetroPT2 ablation + IITK | Rejected | "Concentration depends on data, scaling, representation, sensors, state and time." | "PCA inherently collapses explanations." |
| Robust scaling can increase concentration | MetroPT2 matched-time ablation | Supported in tested settings | "Robust scaling increased concentration in several matched-time comparisons." | "Robust scaling always causes sparse explanations." |
| Flowmeter can dominate explanations under Standard scaling | MetroPT2 full-feature ablation | Supported as detector evidence | "Flowmeter behaved as a dominant detector-evidence channel in several Standard-scaled analyses." | "Flowmeter caused the physical fault." |
| High N_eff is a precursor | MetroPT2 temporal analysis | Rejected | "High N_eff occurs in both anomalous and non-alerting periods." | "Distributed evidence predicts the fault." |
| Explanation spread changes over fault stage | MetroPT2 temporal analysis | Supported descriptively | "Explanation dimensionality changed substantially around reported fault onset." | "This proves causal fault propagation." |
| Universal diffuse→concentrated alert-state transition | calibration-normalized state experiment | Rejected | "No universal alert-state transition was observed." | "AeroXAI discovered a universal fault state machine." |
| PCA generalizes to an independent compressor/acoustic benchmark | IITK | Supported for recording-level seeded-fault novelty detection | "Both PCA variants achieved near-perfect sampled ranking and very high operating recall on IITK." | "PCA is universally optimal for compressors." |
| PCA explanations are intrinsically concentrated | IITK | Rejected | "IITK PCA explanations remained substantially distributed across acoustic feature families." | "PCA always produces one dominant factor." |
| Standard feature-energy ranks IITK faults strongly | IITK | Supported | "Standard feature-energy achieved near-perfect sampled ranking." | "It is therefore the best deployable detector." |
| Robust feature-energy is weak only because of threshold choice | IITK uncertainty | Rejected | "It has both weaker ranking and unstable/conservative calibration." | "Changing the threshold solves the method." |
| PCA calibration is stable on IITK | IITK uncertainty | Supported in this split | "Both PCA variants retained very high recall across calibration resampling." | "PCA thresholds are universally stable." |
| 0/45 healthy false positives means true FPR is zero | IITK uncertainty | Rejected | "No false positives were observed in 45 held-out healthy recordings." | "The true FPR is 0%." |
| sampled AUC=1 means population AUC=1 | IITK uncertainty | Rejected | "Complete separation was observed in this sampled benchmark." | "True AUC is exactly 1." |
| Representation quality = ranking quality | Cross-dataset synthesis | Rejected | "Representation quality also requires explanation behavior and calibration stability." | "Highest AUC automatically means best operational representation." |
| Router V2 retained high-value temporal evidence on future CoBra EVAL data after preregistered within-domain adaptation | REAL CoBra + frozen TCN teacher | Supported in this study | "Router V2 retained 93/93 high-evidence TCN units on the future CoBra EVAL partition." | "Router V2 is universally reliable" or "zero-shot generalization." |
| CoBra validates Router V2 alert-level transport | REAL CoBra | Insufficient evidence | "Only one alert-evidence unit occurred, so alert-level transport was not evaluable." | "Alert coverage was proven to be 100%." |
| Router V2 reduced TCN calls on CoBra | REAL CoBra routing | Supported descriptively | "The frozen router invoked TCN on 78.54% of eligible targets, a 21.46% call reduction." | "TCN calls fell by 21.46%, therefore GPU energy fell by 21.46%." |
| Router V2 met the preregistered CoBra GPU-energy target | MEASURED RTX 4090 GPU-device TCN scoring | Rejected | "Gross GPU-energy reduction was positive in all five repetitions and averaged 9.35%, below the preregistered 50% target." | "The CoBra energy criterion passed." |
| Positive GPU-energy reduction on CoBra demonstrates some compute benefit | MEASURED RTX 4090 GPU-device TCN scoring | Supported narrowly | "The routed condition used less gross GPU energy per logical pass in all five measured repetitions." | "Whole-system AI energy fell by 9.35%." |
| CoBra GPU measurement represents CPU or whole-system compute energy | CoBra energy benchmark | Not supported | "Only GPU-device TCN-scoring energy was measured." | "Total AI energy use fell by 9.35%." |
| MetroPT/IITK/CoBra calibrate the current physics twin | architecture audit | Not supported | "Real detector evidence currently stops at the physical calibration gate." | "The current twin is calibrated from the research datasets." |
| AeroXAI measured factory energy savings on real plant data | current evidence | Not supported | "Physical plant-energy savings remain simulation-only in the current evidence base." | "AeroXAI saved X% in a real factory." |
| Synthetic nominal dispatch saving ≈0.369% | SIMULATED twin/control | Supported in frozen synthetic scenario | "The nominal synthetic dispatch experiment reduced simulated energy by ≈0.369%." | "Real-site saving is 0.369%." |
| Synthetic high-leak energy penalty ≈12.55% | SIMULATED twin | Supported in frozen synthetic scenario | "The modeled high-leak scenario increased simulated energy by ≈12.55%." | "Leak repair saves 12.55% at MetroPT." |
| Robust action is safe | SIMULATED control | Supported only under modeled scenarios/constraints | "The shared first action remained within the modeled safety envelope." | "Guaranteed safe on arbitrary real compressors." |
| AeroXAI autonomously controls equipment | product configuration | False | "Advisory/shadow mode; no PLC/equipment override." | "Autonomous compressor control is deployed." |

## ESIC-ready claims

The following wording is safe for the submission.

> AeroXAI uses explainable anomaly detection on real compressor telemetry and
> explicitly separates detector evidence from physical root-cause claims.

> On MetroPT-3, the frozen PCA detector detected all four documented incidents
> within a predefined timely window; two warnings appeared before reported
> onset and two shortly after onset.

> Follow-up cross-dataset research showed that explanation concentration is not
> an intrinsic property of PCA; it depends on the data representation, scaling,
> available sensors and operating state.

> On an independent IITK compressor/acoustic benchmark, both PCA variants
> achieved complete sampled ROC separation and greater than 99.7% fault recall
> at their frozen healthy-calibration thresholds, while calibration-resampling
> analysis showed substantially stronger operating stability than direct
> feature-energy.

> AeroXAI also evaluates the energy cost of its own AI. In a preregistered
> future CoBra evaluation after within-domain TRAIN/CAL adaptation, the frozen
> evidence-aware router retained all 93 adequate high-evidence TCN cases while
> reducing TCN calls by 21.46%.

> Measured GPU-device TCN-scoring energy was lower in all five routed
> repetitions and fell by 9.35% on average, but this did not meet the
> preregistered 50% target. The result is therefore reported as a failed energy
> criterion rather than tuned post hoc.

> AeroXAI does not allow real anomaly evidence to silently become a physical
> action. The current real-data path stops at a calibration gate when the
> physics bridge is not validated.

> Separately, a physics-based simulated compressor system validates bounded
> state inference, robust shared-action optimization and proof-carrying
> 60-second advisory recommendations.

## Claims that remain research-only

Keep these out of the main product headline unless explicitly labelled as
research or exploratory:

- raw-context TCN pre-onset result;
- MetroPT2 Standard-PCA / feature-energy detector comparisons;
- temporal explanation-state dynamics;
- calibration-normalized state analysis;
- IITK representation comparison as a detector-selection argument;
- IITK uncertainty resampling;
- Router V1 / Router V2 development-set comparisons;
- RCSD router failure diagnosis;
- CoBra Router V2 as a universal routing result;
- CoBra alert-level transport;
- CoBra GPU-energy reduction as a whole-system or plant-energy result.

These studies strengthen the scientific rationale but do not silently alter
the frozen product detector, promote autonomous control, or justify physical
energy-saving claims without a separate promotion and independent validation
path.
