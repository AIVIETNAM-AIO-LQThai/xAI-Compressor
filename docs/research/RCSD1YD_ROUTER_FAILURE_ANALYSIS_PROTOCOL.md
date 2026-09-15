# RCSD-1YD Router Failure Analysis Protocol

Status: **POST-HOC EXPLORATORY PROTOCOL — freeze before new diagnostics**

Parent external-evaluation result commit:

`daa58d4d1a6ecfedafae02f1d0462456d58f6513`

Branch:

`research/rcsd1yd-router-failure-analysis`

Evidence class:

`POST_HOC_EXPLORATORY`

Causal claim:

`false`

## Purpose

The independent RCSD-1YD external evaluation is closed as
`EXTERNAL_TRANSPORT_FAIL`.

This study asks a narrower diagnostic question:

> Why did the frozen eight-feature evidence-aware router lose alignment with
> frozen TCN evidence on RCSD-1YD EVALUATION even though selective inference
> still produced large GPU-device TCN-scoring energy reductions?

RCSD-1YD EVALUATION is no longer an independent test partition after the closed
one-shot experiment. It may now be analyzed for diagnosis, but any result from
this branch is exploratory development evidence only.

Nothing in this branch may retroactively change the closed external-evaluation
classification.

## Closed result that must reproduce

The diagnostic implementation must abort if it cannot reproduce the following
closed EVALUATION facts before producing any new diagnostic output:

- valid TCN rows: 8,820
- CAL q0.90 high-evidence threshold: 0.8942223191261292
- CAL q0.995 alert threshold: 311.1317443847656
- EVALUATION high-evidence bins: 4,647
- EVALUATION alert-evidence bins: 2
- frozen evidence-aware router invocations: 3,270
- frozen evidence-aware high-evidence coverage: 0.39122014202711425
- frozen evidence-aware alert coverage: 0.5
- fixed PCA q90 invocations: 6,084
- fixed PCA q90 high-evidence coverage: 0.9672907251990531

If any of these fail to reproduce exactly within numeric tolerance, stop.

## Frozen diagnostic hypotheses

### H1 — teacher prevalence / distribution shift

The temporal-evidence teacher itself changed materially from CALIBRATION to
EVALUATION.

Diagnostic quantities:

- CAL and EVAL TCN-score count, min, median, q0.50, q0.75, q0.90, q0.95,
  q0.99, q0.995, q0.999, max;
- fraction above the frozen CAL q0.90 threshold in CAL and EVAL;
- fraction above the frozen CAL q0.995 threshold in CAL and EVAL;
- EVAL/CAL ratios where mathematically defined;
- monthly EVAL high-evidence prevalence for October, November, December.

No new teacher threshold may be selected.

### H2 — cheap-feature transport degradation

One or more of the eight frozen cheap-router features changed distribution or
lost their relationship to TCN high evidence.

For each frozen feature:

1. CAL and EVAL descriptive distribution:
   - mean;
   - standard deviation;
   - q0.10;
   - q0.25;
   - median;
   - q0.75;
   - q0.90;
   - min;
   - max.
2. For percentile-valued features, fraction exactly 0 and exactly 1.
3. Univariate relationship with the frozen high-evidence target:
   - raw ROC-AUC;
   - average precision;
   - Spearman correlation with raw TCN score.
4. Report all metrics separately for CAL and EVAL.

These are diagnostics only. No feature is selected, removed, or reweighted in
this study.

### H3 — frozen logistic-combination / calibration transport failure

The eight cheap features may still contain useful information while the frozen
MetroPT-trained logistic combination and probability scale fail to transport.

Freeze these diagnostics:

- frozen router probability distribution on CAL and EVAL;
- ROC-AUC and average precision of frozen router probability versus frozen
  high-evidence target on CAL and EVAL;
- high-evidence enrichment at the already-frozen 0.30 threshold:
  `P(high evidence | routed) / P(high evidence)`;
- precision, recall, and invocation fraction at the frozen 0.30 threshold;
- calibration table using **CAL probability decile boundaries only**:
  - boundaries derived from CAL frozen-router probability using quantiles
    0.10 ... 0.90 with linear interpolation;
  - boundaries are then applied unchanged to both CAL and EVAL;
  - each bin reports row count, mean frozen probability, and observed
    high-evidence fraction.

Do not optimize, sweep, or alter the 0.30 routing threshold.

### H4 — false-negative structure

The closed router missed 2,829 of 4,647 EVALUATION high-evidence bins.

Freeze these groups:

- TP: routed and high evidence;
- FN: not routed and high evidence;
- FP: routed and not high evidence;
- TN: not routed and not high evidence.

For each group report:

- row count;
- frozen router probability median and IQR;
- all eight frozen-feature medians and IQRs;
- PCA-EWMA median and IQR;
- raw TCN-score median and IQR.

The purpose is to identify which frozen signals differ between retained and
missed high-evidence windows. No new classifier is fitted.

## Frozen q90 disagreement analysis

The fixed PCA q90 baseline retained 96.729% of high-evidence EVALUATION bins,
while the frozen evidence-aware router retained 39.122%.

Among EVALUATION high-evidence bins, partition into:

- both q90 and evidence-aware route;
- q90 only;
- evidence-aware only;
- neither.

Report:

- exact counts and fractions;
- frozen router probability median/IQR;
- eight frozen-feature median/IQR;
- raw TCN-score median/IQR.

Also report conditional frozen-router coverage:

- among q90-routed high-evidence bins;
- among q90-not-routed high-evidence bins.

This is a failure diagnostic, not evidence that q90 should be promoted.

## Frozen temporal analysis

For each calendar month in EVALUATION (October, November, December 2022), report:

- valid TCN rows;
- high-evidence prevalence;
- frozen-router invocation fraction;
- high-evidence coverage;
- false-negative count;
- fixed-q90 invocation fraction;
- fixed-q90 high-evidence coverage.

No month-specific threshold or policy is allowed.

## Alert-evidence handling

There are only two EVALUATION alert-evidence bins.

Report their timestamps and the frozen route decisions of the five already
evaluated systems descriptively.

Do not compute alert ROC-AUC, alert average precision, or fit any alert-specific
model because the positive count is too small for useful inferential analysis.

## Explicitly forbidden

This branch must not:

- refit logistic coefficients or intercept;
- fit any new classifier or router;
- tune or sweep the frozen 0.30 probability threshold;
- search q80/q85/q90/etc. for a better PCA routing threshold;
- alter the eight frozen feature definitions;
- invent new router features and score them against RCSD EVALUATION;
- tune the TCN architecture or teacher threshold;
- change CAL references using EVALUATION;
- report RCSD post-hoc results as independent validation;
- revise the closed `EXTERNAL_TRANSPORT_FAIL` result;
- select a Router V2 policy.

A Router V2 study must be a separate branch and must treat RCSD EVALUATION as
development data. Any Router V2 primary validation requires another genuinely
independent compressor dataset.

## Outputs

The implementation stage may create exactly these diagnostic result artifacts:

- `docs/research/rcsd1yd_router_failure_analysis.json`
- `docs/research/RCSD1YD_ROUTER_FAILURE_ANALYSIS.md`

Implementation code/tests will be frozen in a later commit before the
diagnostic result commit whenever practical.

## Interpretation boundary

Permitted conclusion form:

> Post-hoc exploratory analysis indicates that the closed RCSD external failure
> is associated with [observed diagnostic pattern].

Not permitted:

> We validated a new router on RCSD.

Not permitted:

> The failure proves a physical compressor fault mechanism.

The TCN remains a model-evidence teacher, not a physical diagnosis.
