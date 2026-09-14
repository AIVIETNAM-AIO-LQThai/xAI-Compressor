# AeroXAI Research Narrative for ESIC

## One-paragraph version

AeroXAI is not designed as a black-box compressor alarm. Its research program
tests whether an anomaly score is trustworthy enough to enter an operational
reasoning chain. On real MetroPT telemetry, the frozen explainable PCA detector
detects documented compressor incidents and can show which signal groups drove
the alert without claiming physical causality. Follow-up MetroPT2 and IITK
experiments showed that explanation sparsity is not an intrinsic property of
PCA but depends on data geometry, scaling, representation, sensors and
operating state. Independent IITK acoustic testing further showed that PCA can
retain strong ranking while producing distributed evidence and comparatively
stable calibration-derived operating thresholds. AeroXAI therefore treats
anomaly ranking, explanation behavior and calibration stability as separate
proof obligations before evidence is allowed to cross into hypothesis,
physics, optimization and action.

## Judge-friendly research story

### What problem did we discover?

Our first real-data detector worked, but its explanations often concentrated
almost all evidence in one or two signal groups.

Instead of assuming that meant the physical fault was simple, we tested the
explanation mechanism itself.

### What did we test?

We tested:

```text
another MetroPT dataset
different fault labels
removing flow sensors
different scalers
PCA versus direct feature energy
matched explanation timestamps
temporal explanation dynamics
calibration-relative evidence states
an independent acoustic compressor dataset
calibration uncertainty
```

### What did we learn?

The simple hypothesis failed.

PCA itself was not the cause.

Explanation concentration changed with:

```text
data geometry
scaling
representation
sensor availability
operating state
time
```

On IITK acoustic data, PCA achieved very strong sampled fault separation while
its explanations remained distributed across several acoustic feature
families.

### Why does that matter for AeroXAI?

A detector should not be trusted because it has a high AUC alone.

For industrial use, AeroXAI asks three questions:

```text
1. Does it rank abnormal behavior reliably?
2. Can an operator inspect what evidence drove the score?
3. Is the calibration-to-alert boundary statistically stable?
```

Only after those questions are satisfied should anomaly evidence be eligible
to enter the downstream physical-reasoning chain.

### What are we not claiming?

We do not claim:

- detector evidence is physical root cause;
- all failures can be predicted in advance;
- the IITK seeded faults validate real factory savings;
- the current physics twin is calibrated to MetroPT;
- simulated savings are measured site savings.

The current product remains advisory and does not write to compressor or PLC
control.
