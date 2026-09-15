# RCSD-1YD Router Failure Analysis

Evidence class: **POST_HOC_EXPLORATORY**

This analysis diagnoses the already-closed RCSD external transport failure. It is not independent validation and does not alter the `EXTERNAL_TRANSPORT_FAIL` classification.

## Closed-result reproduction

All frozen closed-result guards reproduced before diagnostics ran.

## Teacher shift

- CAL high-evidence prevalence: 10.000%
- EVAL high-evidence prevalence: 52.687%
- CAL alert prevalence: 0.51020%
- EVAL alert prevalence: 0.02268%

## Frozen router transport

- CAL frozen-router ROC-AUC: 0.9481474625627525
- EVAL frozen-router ROC-AUC: 0.4910099463534601
- CAL average precision: 0.7669758300128783
- EVAL average precision: 0.5992113118118378
- CAL enrichment at 0.30: 5.632075471698113
- EVAL enrichment at 0.30: 1.0552176307887302

## False negatives

- EVAL high-evidence false negatives: 2829

## q90 disagreement

- q90-only high-evidence bins: 2737
- evidence-aware-only high-evidence bins: 60
- neither: 92

## Interpretation boundary

These diagnostics may motivate a future Router V2 hypothesis. No Router V2 policy is selected here, and any policy developed using RCSD EVALUATION requires another independent compressor dataset for primary validation.
