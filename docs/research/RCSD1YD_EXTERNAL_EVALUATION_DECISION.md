# RCSD-1YD External Evaluation Decision

## Status

**EXTERNAL TRANSPORT FAIL**

The one-shot RCSD-1YD evaluation was executed only after the adaptation freeze
and execution harness were committed. The frozen evidence-aware router failed
the preregistered evidence-retention criteria, despite passing the preregistered
GPU-device TCN-scoring energy criterion.

No post-EVALUATION tuning is allowed for this study.

## Frozen evaluation scope

Dataset:
- RCSD-1YD refinery centrifugal compressor
- EVALUATION source rows: 8,832
- valid causal TCN target rows: 8,820

Frozen CAL-derived temporal-evidence thresholds:
- high evidence: CAL q0.90
- alert evidence: CAL q0.995

Observed EVALUATION teacher positives:
- high-evidence bins: 4,647 / 8,820 = 52.687%
- alert-evidence bins: 2 / 8,820 = 0.023%

The two alert bins make alert coverage numerically defined under the frozen
adequacy rule, but the alert result is statistically sparse. The high-evidence
criterion alone is sufficient to establish failure.

## Primary frozen evidence-aware router result

Frozen evidence-aware router:
- TCN invocation fraction: 37.075%
- high-evidence coverage: 39.122%
- alert-evidence coverage: 50.000%
- mean gross GPU-device TCN-scoring energy reduction: 57.003%
- gross energy reduction positive in all 5/5 repetitions
- mean incremental GPU-device reduction: 58.590%

Frozen success criteria:
- high-evidence coverage >= 80%
- alert-evidence coverage >= 80%
- mean gross GPU energy reduction >= 50%
- gross energy reduction positive in all 5 repetitions

Outcome:
- high-evidence coverage: **FAIL**
- alert-evidence coverage: **FAIL**
- mean gross GPU energy reduction: **PASS**
- positive gross reduction 5/5: **PASS**

Final classification:

`EXTERNAL_TRANSPORT_FAIL`

## Frozen baseline comparison

| System | TCN invocation | High-evidence coverage | Alert coverage | Mean gross GPU reduction |
|---|---:|---:|---:|---:|
| Always-on TCN | 100.000% | 100.000% | 100.000% | 0% reference |
| Fixed PCA q90 | 68.980% | 96.729% | 100.000% | 34.315% |
| Fixed PCA q95 | 40.964% | 70.519% | 100.000% | 57.801% |
| Fixed PCA q99 | 0.170% | 0.301% | 100.000% | 93.273% |
| Frozen evidence-aware router | 37.075% | 39.122% | 50.000% | 57.003% |

No frozen selective-compute system satisfies all preregistered primary
constraints simultaneously.

The q90 baseline retains temporal evidence well, but does not reach the
required >=50% mean gross GPU-energy reduction.

The q95 baseline reaches the energy target and retains all two alert bins, but
its high-evidence coverage is only 70.519%, below the frozen 80% criterion.

The q99 baseline saves the most measured GPU TCN-scoring energy but nearly
eliminates high-evidence retention.

## Mechanistic interpretation

The RCSD temporal-evidence distribution shifted strongly relative to
CALIBRATION:

- the CAL q0.90 threshold was exceeded by 52.687% of EVALUATION TCN windows;
- the CAL q0.995 threshold was exceeded by only 2 EVALUATION windows.

This is consistent with a substantial change in the shape of the temporal-error
distribution, not merely a simple uniform scale shift.

The frozen evidence-aware router invoked TCN on 37.075% of windows but covered
39.122% of high-evidence windows. Relative to its invocation rate, this is only
modest enrichment. The cheap routing representation therefore did not preserve
the relationship to TCN evidence strongly enough under this cross-domain shift.

This result does not imply a physical fault rate of 52.687%. TCN evidence is a
model-error/evidence signal, not a physical diagnosis.

## Energy interpretation

The energy mechanism transported:

- frozen evidence-aware router mean gross reduction: 57.003%
- per-repetition gross reductions:
  - 57.524%
  - 58.213%
  - 51.766%
  - 59.999%
  - 57.511%
- positive in 5/5 repetitions

Measured scope remains:

**GPU-device TCN scoring only**

Not measured:
- CPU/router energy
- whole-system compute energy
- physical compressor energy
- carbon impact

Therefore this study supports the narrower statement that selective routing can
materially reduce GPU TCN-scoring energy on RCSD-1YD, but the frozen
evidence-aware policy did not retain enough temporal evidence to pass external
transport.

## Scientific conclusion

The external study demonstrates a real trade-off rather than a successful
transport result:

> The frozen evidence-aware router preserved the compute-saving mechanism on a
> new refinery centrifugal-compressor dataset, but its evidence-selection
> mechanism did not generalize sufficiently. External transport therefore
> failed under the preregistered joint evidence-retention and GPU-energy
> criteria.

This is stronger evidence than further MetroPT tuning because it identifies a
specific limitation on genuinely new chronological compressor data.

## Required decision

Close the frozen RCSD-1YD external-evaluation study as:

`EXTERNAL_TRANSPORT_FAIL`

Do not:
- change the frozen router threshold;
- refit logistic coefficients;
- alter router features using RCSD EVALUATION;
- change TCN thresholds using RCSD EVALUATION;
- promote q90/q95/q99 based on this EVALUATION as a newly validated policy;
- rerun alternate policies and present them as independent external validation.

Post-hoc failure analysis on RCSD-1YD is allowed only as explicitly exploratory
diagnosis. Any new policy derived from RCSD-1YD EVALUATION requires a new
independent chronological compressor dataset for validation.
