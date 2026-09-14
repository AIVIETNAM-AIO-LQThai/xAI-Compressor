# RCSD-1YD Adaptation Decision

## Status

**TRAIN/CAL ADAPTATION PASS — freeze before EVALUATION**

The RCSD-1YD source identity and frozen schema passed ingestion:

- source MD5: `9ebf0bf829d2567f279f385523a765c5`
- TRAIN rows: 17,372
- CALIBRATION rows: 8,832
- EVALUATION rows: 8,832
- TRAIN dropped non-finite rows: 0
- CALIBRATION dropped non-finite rows: 0
- TRAIN gaps > 30 minutes: 1
- CALIBRATION gaps > 30 minutes: 0

No EVALUATION sensor summary, TCN metric, router metric, or energy metric was
computed during adaptation.

## Dataset-specific TCN

Frozen adaptation result:

- fit samples: 13,878
- chronological validation samples: 3,470
- best epoch: 19
- epochs run: 24
- best TRAIN-tail validation MSE: `32.89568440110264`
- CALIBRATION score rows: 8,820
- CAL q0.90 evidence threshold: `0.8942223191261292`
- CAL q0.995 alert-evidence threshold: `311.1317443847656`
- checkpoint SHA256:
  `a4ebea11a3f0a1a4de3afc633a7dbee6a9e612740139636b39c77d7d421c8c50`

The large separation between CAL q0.90 and q0.995 indicates a strongly
heavy-tailed temporal-error distribution. This is not a protocol failure:
the thresholds are finite, ordered, and were derived from CALIBRATION only.
It must be preserved rather than tuned after seeing EVALUATION.

The large TRAIN-tail validation MSE relative to the CAL q0.90 threshold is
also recorded as evidence of strong chronological heterogeneity/outlier
influence in the new domain. It is not grounds for post-hoc architecture or
hyperparameter changes.

## PCA / cheap-signal adaptation

- PCA components retained at 95% variance: 3
- CAL router-reference rows: 8,820
- PCA-EWMA q0.90: `2.866995410407013`
- PCA-EWMA q0.95: `3.4541060736104505`
- PCA-EWMA q0.99: `9.048523581056804`

The three-component PCA is dataset-specific and reflects the covariance
structure of the 25 RCSD process channels. It does not alter the frozen
logistic router.

## Frozen parent router identity

- C: 10.0
- class_weight: None
- probability threshold: 0.30
- frozen identity SHA256:
  `569acf048316327e36478066566b54225e9dc2bcae54c84d21ea00f9fc1bc04a`

No logistic coefficient, intercept, final router-feature scaler, probability
threshold, or router feature definition was refit on RCSD-1YD.

## Decision

**Commit the adaptation freeze before computing any EVALUATION metric.**

After this commit, the EVALUATION partition may be opened exactly once under
the preregistered evaluation protocol.

Do not:

- retune the TCN from EVALUATION behavior;
- change CAL q0.90 or q0.995 thresholds;
- change PCA/support calibration references;
- refit or rescale the logistic router;
- change the 0.30 routing threshold;
- alter missing/gap handling;
- use EVALUATION to select among alternate policies.

A failure on RCSD-1YD EVALUATION must be recorded as external transport
failure rather than rescued by post-hoc tuning.
