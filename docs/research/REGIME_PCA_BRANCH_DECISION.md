# Regime-Conditioned Standard-PCA Research Branch Decision

## Status

**Decision: CLOSE / DO NOT PROMOTE**

Branch: `research/regime-conditioned-pca`

The regime-conditioned Standard-PCA line of research is closed for model
selection on the current MetroPT-3 reused TEST evidence.

No additional router, scaler, PCA, smoothing, persistence, feature-removal,
coverage filter, or support-gating candidate should be selected using this TEST
set.

The frozen Robust-PCA detector remains the leading operational research
reference unless and until new independent evidence supports a replacement.

## What the branch established

### 1. The apparent two-dimensional Robust-PCA geometry was partly a scaling artifact

Robust scaling gave extreme leverage to several small-IQR engineered features.
Under Standard scaling, the representation required substantially more
principal components and showed richer geometry.

Therefore the original "extreme low dimensionality" interpretation cannot be
treated as intrinsic compressor geometry.

### 2. Operating regimes are geometrically meaningful

Under Standard scaling, current/reservoir operating regimes showed coherent
local neighborhoods and strongly different PCA tangent directions.

Regime-conditioned PCA materially improved held-out normal reconstruction
inside the original TRAIN/CALIBRATION support.

This is a valid representation finding.

### 3. Better representation did not imply better operational calibration

The regime-conditioned Standard-PCA benchmark improved ranking and pre-onset
behavior relative to global Standard-PCA, but failed strict promotion against
the frozen Robust-PCA reference because false-alert burden and calibration
uncertainty were substantially worse.

### 4. Cross-regime EWMA state carryover was a real mechanism

A single pooled EWMA carried anomaly-score history across hard regime changes.

Resetting EWMA at regime transitions substantially reduced calibration
autocorrelation and bootstrap threshold uncertainty.

This established a real temporal-state contamination mechanism.

### 5. Fixing EWMA carryover did not produce a better detector

The transition-reset EWMA candidate improved calibration stability but worsened
TEST false-alert burden, reduced PR-AUC, and reduced pre-onset incident recall.

Therefore transition-reset EWMA is not promoted.

### 6. Alert fragmentation was real but was not the upstream cause

Many false episode starts used persistence windows spanning regime boundaries,
and transition-reset smoothing produced shorter, more fragmented alert
episodes.

However the larger failure existed before persistence: CALIBRATION-to-TEST raw
score transport was severely broken.

### 7. The dominant transport problem was Standard-scaled feature stationarity

Global Robust-PCA transported comparatively well.

Both global Standard-PCA and regime-conditioned Standard-PCA showed severe
CALIBRATION-to-TEST score inflation.

The largest detector-evidence shifts involved TP2, DV-pressure, and
pressure-switch engineered features.

### 8. TP2 and DV-pressure shifts were not merely isolated minimum-statistic artifacts

Extreme TP2 minimum rows generally also had shifted TP2 mean values.

Extreme DV-pressure minimum rows generally also had shifted DV-pressure mean
values.

Therefore the analog transport failure cannot be explained only by one-bin
minimum-feature spikes.

### 9. Digital pressure-switch novelty was strongly coupled to incomplete aggregation windows

Pressure-switch transition and active-ratio support violations became extremely
common in underfilled 5-minute windows but were rare in full-coverage windows.

This component is primarily a data-quality / aggregation-support issue and
should not be treated as evidence for a new physical operating regime.

### 10. Analog TP2 / DV-pressure novelty survives full coverage

After conditioning descriptively on full-coverage / complete-sample windows:

- analog TRAIN-support violations still occur;
- the TP2/DV-pressure joint novelty pattern remains;
- those rows still have extremely large regime-PCA residual scores;
- the dominant full-coverage novelty is concentrated especially in the
  `high_current_only` regime but also appears in `high_current_high_pressure`.

Therefore incomplete coverage is not sufficient to explain the analog shift.

### 11. The existing four-regime router does not represent all later operating support

The current current/reservoir q80 router maps unsupported analog states into
existing regimes.

However those analog states are absent or essentially absent from
TRAIN/CALIBRATION.

A new local PCA for that state therefore cannot be legitimately learned from
the currently frozen training evidence.

Using reused TEST rows to define and train a new "normal" regime would convert
the benchmark set into training/model-selection data.

That is not allowed for a credible promotion claim.

## Final model-selection decision

### Rejected for promotion

- global Standard-PCA;
- regime-conditioned Standard-PCA;
- regime-conditioned Standard-PCA + transition-reset EWMA;
- any post-hoc router expansion derived from MetroPT-3 TEST;
- any post-hoc feature removal or support threshold tuned to MetroPT-3 TEST.

### Retained as scientific findings

- Standard vs Robust scaling materially changes inferred PCA geometry;
- compressor behavior is locally multi-regime;
- regime-local PCA improves in-support reconstruction;
- hard-regime temporal smoothing can create cross-regime state contamination;
- Standard-scaled regime PCA is vulnerable to later analog operating-support
  shift;
- digital pressure-switch engineered features are sensitive to aggregation
  completeness;
- TP2/DV-pressure analog support shift persists under full-coverage
  conditioning.

### Leading operational research reference

The frozen Robust-PCA detector remains the leading reference because the
regime-conditioned Standard-PCA research line has not demonstrated robust
out-of-support transport or strict operational dominance.

This does not prove Robust-PCA is universally optimal.

## Required evidence before reopening this model line

A future regime/state-conditioned candidate requires **new independent data**,
not further reuse of the current MetroPT-3 TEST set.

The minimum acceptable next dataset should contain:

1. chronological compressor telemetry not used in the current TEST-driven
   diagnostics;
2. confirmed or defensible normal-operation intervals;
3. full or explicitly audited sampling coverage;
4. examples of the TP2/DV-pressure high-current operating state observed in
   later MetroPT data;
5. enough repeated cycles to estimate state occupancy and calibration
   uncertainty;
6. incident/fault intervals held out from all state-definition and
   calibration decisions.

Only after obtaining such evidence should a new study preregister one of:

- richer operating-state routing;
- learned state clustering fitted only on normal training data;
- mixture/local PCA;
- state-aware calibration;
- open-set / unsupported-state abstention.

## Claim discipline

The current evidence supports:

> Regime-conditioned Standard-PCA improved in-support normal reconstruction and
> exposed meaningful local operating geometry, but it did not generalize
> robustly to later MetroPT operating support. The failure was driven primarily
> by Standard-scaled analog TP2/DV-pressure distribution shift, with a separate
> aggregation-quality effect in digital pressure-switch features.

The current evidence does **not** support:

- claiming TP2 or DV-pressure is the physical root cause of a compressor fault;
- claiming all background TEST rows are healthy;
- treating TEST-discovered support states as legitimate training regimes;
- claiming transition-reset smoothing is operationally superior;
- claiming regime-conditioned PCA dominates the frozen Robust-PCA detector;
- promoting any new router without independent data.

## Research stop condition

The current branch has reached its stop condition:

**No further model-selection iteration on reused MetroPT-3 TEST.**

Next action is independent-data acquisition / construction of a genuinely
unseen confirmation split before reopening the regime/state-conditioned model
line.
