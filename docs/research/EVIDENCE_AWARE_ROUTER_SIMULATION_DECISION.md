# Evidence-Aware Router — Simulated Mechanism Decision

## Status

**SIMULATED MECHANISM PASS — with weak support-tail sensitivity**

This result is a mechanism diagnostic only. It is not independent validation,
does not satisfy the primary generalization claim, and does not support claims
about physical compressor-energy savings, whole-system compute energy, or
carbon savings.

Frozen policy commit:

`aa3863080493c19483d536a2f9d2da1613a651b7`

Frozen router:

- logistic regression
- C = 10.0
- class_weight = None
- probability threshold = 0.30

No router refit, feature change, or threshold tuning was performed.

## Baseline reproduction

The frozen development-policy invocation fractions reproduced exactly before
synthetic perturbations were evaluated:

| Dataset | Frozen baseline invocation |
|---|---:|
| MetroPT-3 | 16.282% |
| MetroPT2 | 19.477% |

All baseline reproduction checks passed.

## Feature-location shift

Synthetic positive location shift was applied only to the frozen continuous
analog / pressure-related feature scope.

### MetroPT-3

| Shift severity | Invocation |
|---|---:|
| baseline | 16.282% |
| +0.25 TRAIN sigma | 17.435% |
| +0.50 TRAIN sigma | 19.020% |
| +1.00 TRAIN sigma | 23.631% |

Maximum change from baseline: **+7.349 percentage points**.

### MetroPT2

| Shift severity | Invocation |
|---|---:|
| baseline | 19.477% |
| +0.25 TRAIN sigma | 20.379% |
| +0.50 TRAIN sigma | 23.445% |
| +1.00 TRAIN sigma | 31.650% |

Maximum change from baseline: **+12.173 percentage points**.

The preregistered nondecreasing mechanism check passed on both datasets.

## Feature-scale shift

### MetroPT-3

| Scale factor | Invocation |
|---|---:|
| baseline | 16.282% |
| 1.10 | 17.003% |
| 1.25 | 19.452% |
| 1.50 | 25.504% |

Maximum change from baseline: **+9.222 percentage points**.

### MetroPT2

| Scale factor | Invocation |
|---|---:|
| baseline | 19.477% |
| 1.10 | 20.559% |
| 1.25 | 23.986% |
| 1.50 | 28.494% |

Maximum change from baseline: **+9.017 percentage points**.

The preregistered nondecreasing mechanism check passed on both datasets.

## Support-tail inflation

### MetroPT-3

| Tail factor | Invocation |
|---|---:|
| baseline | 16.282% |
| 1.25 | 16.282% |
| 1.50 | 16.282% |
| 2.00 | 16.427% |

Maximum change from baseline: **+0.144 percentage points**.

### MetroPT2

| Tail factor | Invocation |
|---|---:|
| baseline | 19.477% |
| 1.25 | 19.477% |
| 1.50 | 19.567% |
| 2.00 | 19.567% |

Maximum change from baseline: **+0.090 percentage points**.

The frozen preregistered check was nondecreasing, so this stress family
formally passes. However, the response is practically very small. Therefore
the result should be described as **weak support-tail sensitivity**, not as
strong adaptive escalation under every type of support shift.

No post-result retuning is allowed.

## Operating-mixture counterfactual

No monotonic direction was preregistered for operating-mixture shift.

### MetroPT-3

| Counterfactual mixture | Invocation |
|---|---:|
| low-dominant | 6.169% |
| balanced | 14.174% |
| high-dominant | 22.637% |

### MetroPT2

| Counterfactual mixture | Invocation |
|---|---:|
| low-dominant | 12.308% |
| balanced | 21.300% |
| high-dominant | 29.463% |

These are counterfactual aggregate compute-demand estimates using frozen
regime-conditional routing rates. They are not new chronological observations
and are not evidence of fault-detection performance.

## Router CPU wall-time diagnostic

Measured scope:

- frozen 8-feature StandardScaler transform
- logistic probability
- fixed 0.30 threshold decision

The measured amortized batch wall time was approximately:

`0.105 microseconds / row`

with sample SD approximately:

`0.003 microseconds / row`

This excludes upstream PCA and support-feature construction and must not be
reported as end-to-end router latency.

CPU energy remains **UNKNOWN**.

Whole-system energy remains **UNKNOWN**.

## Scientific interpretation

The frozen router shows the intended qualitative compute-allocation response
under two broad synthetic distribution-shift families:

- stronger location shift -> more TCN invocation;
- stronger scale shift -> more TCN invocation.

Support-tail inflation is nondecreasing but nearly insensitive under the
frozen percentile-based representation.

Operating-mixture counterfactuals show that aggregate compute demand can vary
substantially with the operating-state mixture, but no causal or performance
claim follows from that result.

## Decision

**Record the SIMULATED mechanism result and proceed to a genuinely new
chronological compressor-data protocol.**

Do not:

- retune the frozen router;
- alter the support-tail feature after observing this result;
- call these simulations independent validation;
- use this result to satisfy the primary evidence-coverage or GPU-energy
  criteria;
- infer physical compressor-energy or carbon savings.

The next scientific step is to freeze provenance, sensor mapping, chronology,
missing-data handling, TRAIN/CALIBRATION/EVALUATION boundaries, and model
adaptation semantics for a genuinely new compressor dataset before opening
its EVALUATION partition.
