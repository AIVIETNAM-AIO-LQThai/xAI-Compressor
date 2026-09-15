# RCSD-1YD Router Failure Analysis Decision

## Status

**POST-HOC EXPLORATORY DIAGNOSIS COMPLETE**

The closed RCSD-1YD external evaluation remains:

`EXTERNAL_TRANSPORT_FAIL`

The reproduction guard passed before any diagnostic interpretation. This branch
does not revise the closed result and does not select Router V2.

## Primary diagnosis

The external failure is best explained by a combination of:

1. **strong teacher-distribution / prevalence shift**, and
2. **failure of the frozen logistic feature combination to transport**,

with **selective cheap-feature degradation**, rather than a wholesale failure
of the cheap PCA-based representation.

The evidence does **not** support the stronger claim that all cheap signals
became uninformative on RCSD-1YD.

## H1 — Teacher-distribution shift: SUPPORTED

Frozen CAL q0.90 high-evidence prevalence:

- CALIBRATION: 10.000%
- EVALUATION: 52.687%
- EVAL/CAL prevalence ratio: 5.269x

TCN-score distribution:

- CAL median: 0.5235
- EVAL median: 0.9056
- CAL q0.90: 0.8942
- EVAL q0.90: 1.0971

The extreme tail changed differently:

- CAL q0.995: 311.1267
- EVAL q0.995: 1.3477
- CAL alert prevalence: 0.51020%
- EVAL alert prevalence: 0.02268%
- EVAL alert positives: 2

This is evidence of a major change in the shape of the temporal-model error
distribution. It is not evidence that 52.687% of RCSD operation represented
physical compressor faults.

## H2 — Cheap-feature transport degradation: PARTIALLY SUPPORTED

Several cheap features retained strong relationship to frozen high-TCN evidence
on EVALUATION:

- `pca_ewma_percentile`
  - CAL ROC-AUC: 0.9762
  - EVAL ROC-AUC: 0.9272
  - EVAL AP: 0.9256
  - EVAL Spearman with raw TCN score: 0.8523
- `recent_q95_hit_fraction`
  - EVAL ROC-AUC: 0.8427
  - EVAL AP: 0.8244
  - EVAL Spearman: 0.7248
- `recent_q90_hit_fraction`
  - EVAL ROC-AUC: 0.8090
  - EVAL AP: 0.7487
  - EVAL Spearman: 0.7119

However, important support-related features degraded sharply:

- `feature_support_percentile`
  - CAL ROC-AUC: 0.9421
  - EVAL ROC-AUC: 0.4625
  - CAL Spearman: 0.7052
  - EVAL Spearman: -0.0202
- `combined_support_percentile`
  - CAL ROC-AUC: 0.8773
  - EVAL ROC-AUC: 0.4737
  - CAL Spearman: 0.6166
  - EVAL Spearman: 0.0044
- `pca_ewma_volatility_percentile`
  - CAL ROC-AUC: 0.8805
  - EVAL ROC-AUC: 0.6527

Therefore the cheap representation did not fail uniformly. Transport failure
was feature-specific.

## H3 — Frozen logistic combination / calibration transport failure: STRONGLY SUPPORTED

Frozen router performance changed from:

- CAL ROC-AUC: 0.9481
- CAL AP: 0.7670
- CAL enrichment at threshold 0.30: 5.632x

to:

- EVAL ROC-AUC: 0.4910
- EVAL AP: 0.5992
- EVAL enrichment at threshold 0.30: 1.055x

At the frozen threshold:

- EVAL invocation fraction: 37.075%
- EVAL precision: 55.596%
- EVAL recall: 39.122%
- EVAL high-evidence prevalence: 52.687%

The router therefore provided almost no useful ranking advantage over the
EVALUATION base rate, even though several individual cheap features remained
highly predictive.

This is the strongest diagnostic evidence that the learned MetroPT logistic
combination/probability mapping did not transport to RCSD-1YD.

## H4 — False-negative structure: SUPPORTED AS A FAILURE DESCRIPTION

EVALUATION confusion groups:

- TP: 1,818
- FN: 2,829
- FP: 1,452
- TN: 2,721

The frozen router missed 2,829 of 4,647 high-evidence bins.

Among high-evidence bins:

- both q90 and evidence-aware router: 1,758
- q90 only: 2,737
- evidence-aware only: 60
- neither: 92

Thus 58.898% of all high-evidence bins were captured by q90 but rejected by the
frozen evidence-aware router.

Evidence-aware coverage was nearly identical inside and outside q90-routed
high-evidence subsets:

- within q90-routed high evidence: 39.110%
- within q90-not-routed high evidence: 39.474%

This supports the interpretation that the evidence-aware router lost the useful
ranking structure present in the PCA-EWMA signal rather than simply applying a
more aggressive version of q90 routing.

## Representation-mismatch hypothesis: NOT PRIMARY

A deep representation failure is not the main explanation supported by these
diagnostics.

Reasons:

- `pca_ewma_percentile` retained ROC-AUC 0.9272 on EVALUATION;
- recent q90/q95 history features also retained substantial predictive value;
- fixed PCA q90 retained 96.729% of high-evidence bins.

Therefore RCSD still contained useful evidence in the frozen cheap PCA-based
signals. The principal failure occurred in how the frozen router combined
transporting and non-transporting features under a shifted temporal-evidence
distribution.

This does not prove that the eight-feature representation is optimal or
sufficient for future routing.

## Temporal structure

Failure was not confined to one month:

### October 2022

- high-evidence prevalence: 45.783%
- evidence-aware invocation: 20.513%
- evidence-aware high coverage: 15.623%
- false negatives: 1,145
- q90 high coverage: 95.210%

### November 2022

- high-evidence prevalence: 85.486%
- evidence-aware invocation: 50.035%
- evidence-aware high coverage: 49.838%
- false negatives: 1,235
- q90 invocation: 99.514%
- q90 high coverage: 99.919%

### December 2022

- high-evidence prevalence: 27.823%
- evidence-aware invocation: 41.028%
- evidence-aware high coverage: 45.773%
- false negatives: 449
- q90 high coverage: 89.734%

The temporal-evidence prevalence and compute demand changed substantially across
months. November is especially important: a conservative q90 policy effectively
approached always-on inference, demonstrating that evidence retention and
compute saving are regime dependent.

No month-specific policy is selected here.

## Alert evidence

Only two EVALUATION alert-positive bins existed:

- 2022-12-12 11:45 — frozen evidence-aware router routed the bin
- 2022-12-12 12:00 — frozen evidence-aware router missed the bin

All three fixed PCA q90/q95/q99 routes included both alert bins.

Because there are only two positives, no alert-specific classifier inference,
AUC claim, or policy selection is warranted.

## Scientific conclusion

The most defensible post-hoc conclusion is:

> RCSD-1YD exposed a cross-domain failure of the frozen MetroPT-trained
> evidence-aware logistic routing function, not a general failure of cheap
> PCA-based evidence signals. The TCN evidence distribution shifted strongly,
> support-related router features lost transportability, while PCA-EWMA and
> recent-threshold-history features remained informative. The frozen logistic
> combination consequently collapsed from strong CAL discrimination to
> approximately chance-level EVALUATION ranking.

This is a diagnostic result, not independent validation.

## Decision

Close this failure-analysis branch after committing the diagnostic artifacts.

Do not:

- tune the 0.30 threshold on RCSD EVALUATION;
- refit the current logistic router and call it validated;
- promote q90/q95/q99 as externally validated policies;
- add features post hoc and score them as if independent;
- revise the closed `EXTERNAL_TRANSPORT_FAIL` result.

## Next study

A separate Router V2 development study is justified.

RCSD-1YD EVALUATION may now be treated as **development data** in that new
study, together with MetroPT development data, because its independent-test role
has been consumed.

A Router V2 hypothesis should focus on transport robustness rather than simply
adding model capacity. Candidate research directions may include:

- reducing dependence on dataset-specific support features;
- preserving strongly transporting PCA-EWMA and temporal-hit information;
- explicit domain/support-shift awareness;
- calibration strategies that do not assume stable teacher prevalence;
- conservative fallback behavior when routing confidence is unsupported.

Any Router V2 primary claim requires a **new, genuinely independent compressor
dataset** that has not influenced Router V2 design, fitting, thresholding, or
model selection.
