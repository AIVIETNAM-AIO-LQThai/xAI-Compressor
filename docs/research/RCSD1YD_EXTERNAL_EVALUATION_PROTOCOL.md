# RCSD-1YD External Evaluation Protocol

Status: **PREREGISTRATION — SOURCE DATA NOT YET OPENED**

Parent simulation result:

`03f16e15c63f1292ed8fcad345ce38547d66bb23`

Frozen evidence-aware router:

`aa3863080493c19483d536a2f9d2da1613a651b7`

## Purpose

Perform a one-shot external chronological evaluation of the already frozen
evidence-aware compute router on a substantially different real compressor:
the RCSD-1YD refinery centrifugal-compressor dataset.

This stage is not a new router-development stage.

No logistic coefficients, router feature definitions, router scaler, or routing
threshold may be fitted or changed using RCSD-1YD.

## Dataset identity

Dataset:

**Refinery Compressor Sensor Data, One-Year Dataset (RCSD-1YD)**

Dataset DOI:

`10.5281/zenodo.14866092`

Descriptor:

`10.1109/IEEEDATA.2025.3571011`

Expected archived file:

`Centrifugal compressor _2022.xlsx`

Expected MD5:

`9ebf0bf829d2567f279f385523a765c5`

The public descriptor states that the dataset contains 25 process-sensor
channels plus timestamp, covers the full 2022 calendar year, and is sampled
every 15 minutes. One hour is reported missing on 27 March 2022 because of the
summer-time transition.

The source file must not be used if its MD5 does not match the frozen identity.

## Frozen sensor schema

Use all 25 descriptor-listed sensor tags and no additional raw features.

The raw model input therefore has 25 dimensions.

The dataset's published equipment thresholds are metadata only; they are not
inputs to the TCN, PCA, support model, or logistic router.

## Chronological split

Freeze calendar boundaries before opening the source file:

TRAIN:

`2022-01-01 00:00:00 <= t < 2022-07-01 00:00:00`

CALIBRATION:

`2022-07-01 00:00:00 <= t < 2022-10-01 00:00:00`

EVALUATION:

`2022-10-01 00:00:00 <= t <= 2022-12-31 23:45:00`

No shuffle.

The split is calendar-based rather than row-count-based so the split cannot
move in response to missing data.

## Missingness and gaps

Do not impute source sensor values.

Any row with a missing or non-finite value in any of the 25 frozen channels is
removed.

After removal, a timestamp difference greater than 30 minutes creates a new
contiguous sequence segment.

TCN windows may not cross those boundaries.

PCA causal EWMA resets at gaps greater than 30 minutes.

Rolling cheap-router history also may not bridge a gap boundary.

## Dataset-specific TCN

The TCN is a new dataset-specific temporal evidence model, not the model being
validated.

Freeze the previous topology/training family:

- hidden channels: 32, 32, 32;
- kernel: 3;
- dilations: 1, 2, 4;
- dropout: 0.10;
- sequence length: 12 native bins;
- native cadence: 15 minutes;
- resulting history: 180 minutes;
- causal next-step prediction;
- MSE;
- AdamW;
- lr 0.001;
- weight decay 0.0001;
- batch 256;
- max 50 epochs;
- patience 5;
- seed 20260915;
- TRAIN internal split: first 80% fit, last 20% validation.

Input dimension is 25 because it is fixed by the source schema, not selected by
performance.

No architecture or hyperparameter search is allowed.

## Dataset-specific PCA and support transforms

Fit Robust-PCA on TRAIN only.

Retain 95% variance.

Fit the support StandardScaler on TRAIN only.

CALIBRATION may set the dataset-specific reference distributions needed to
turn cheap signals into normalized router inputs:

- PCA-EWMA percentile;
- PCA-EWMA delta median/IQR;
- PCA-EWMA volatility percentile;
- feature-support percentile;
- latent-support percentile;
- q90/q95 recent-hit thresholds.

This is calibration of normalized input coordinates, not refitting the router.

## Frozen logistic router

Apply exactly the policy frozen in the parent study:

- C = 10;
- class weight = None;
- threshold = 0.30;
- same eight cheap-router features;
- same frozen final router-feature StandardScaler;
- same coefficients;
- same intercept.

No RCSD-1YD row may change these quantities.

## Adaptation freeze before EVALUATION

After TRAIN/CALIBRATION processing, create a committed adaptation artifact
before computing EVALUATION metrics.

That artifact must include:

- source MD5;
- exact matched schema;
- TRAIN/CAL row counts;
- EVALUATION row count and calendar bounds only;
- missing/gap counts for TRAIN/CAL;
- TCN checkpoint hash;
- TCN training-validation result;
- CAL q90/q995 TCN evidence thresholds;
- PCA/support reference parameters or hashes;
- identity of the frozen logistic router.

No EVALUATION sensor distribution, router result, TCN result, or energy result
may appear in that pre-evaluation artifact.

## One-shot EVALUATION

Evaluate five systems:

1. always-on TCN;
2. fixed PCA q90;
3. fixed PCA q95;
4. fixed PCA q99;
5. frozen evidence-aware router.

Define temporal evidence using CAL thresholds:

High evidence:

`EVAL raw TCN score >= CAL q0.90`

Alert evidence:

`EVAL raw TCN score >= CAL q0.995`

The frozen router passes only if all are true:

- high-evidence bin coverage >= 80%;
- alert-evidence bin coverage >= 80%;
- mean gross GPU-device TCN scoring energy reduction >= 50% versus always-on;
- gross reduction is positive in all five measured repetitions.

There is no weighted rescue score.

## Energy boundary

Use the same in-process NVML protocol:

- 100 ms sampling;
- two warm-ups;
- five repetitions;
- paired 10-second idle;
- minimum 20-second metered workload.

The measured quantity remains GPU-device TCN scoring energy only.

CPU/router energy: UNKNOWN.

Whole-system energy: UNKNOWN.

Physical compressor energy: NOT MEASURED.

## Interpretation

A PASS supports:

> Cross-domain external transport of the frozen evidence-aware compute-routing
> mechanism from MetroPT development data to a real refinery centrifugal
> compressor.

A PASS does not establish:

- universal compressor generalization;
- physical fault diagnosis;
- physical energy savings;
- whole-system AI energy savings;
- carbon savings.

A FAIL is recorded as external transport failure. The EVALUATION partition
must not then be used to retune this frozen router.
