# Syrus case files

Deterministic evidence for Syrus, built from frozen AeroXAI artifacts. Nothing here retrains, retunes or re-thresholds the detector.

```bash
python -m scripts.build_syrus_cases          # rebuild docs/syrus/cases/
python -m scripts.build_syrus_cases --check  # fail if the committed files differ from a fresh build
```

## Inputs

- Frozen reports: `docs/xai_report.json`, `docs/xai_verification_report.json`, `docs/detector_benchmark.json`, `docs/proof_manifest.json`.
- Configs: `configs/metropt.yaml`, `configs/detection.yaml`, `configs/evidence_action_bridge.yaml`.
- Syrus configs: `configs/syrus_cases.yaml` (opaque case IDs, display and evidence rules), `configs/syrus_hypotheses.yaml` (unranked candidate catalog), `configs/syrus_discriminators.yaml` (separating observations, statuses, follow-up questions). The last two are drafts that need domain review.

`manifest.json` records the content hash of every input. JSON inputs are hashed as canonical JSON and text inputs with LF line endings, so the hashes do not depend on Git line-ending settings.

## Files

One pair per case, listed in the manifest by alert time:

- `{case_id}.retro.json`: every record, with full source pointers (`file`, `path`, `sha256`). Paths use `[field=value]` selectors, so they do not depend on list order.
- `{case_id}.asof.json`: only records visible at alert time. Source pointers keep the file and hash but drop the path, which would reveal the incident number. It has no label, onset, timing, case status, benchmark or document content.

Each record has `id`, `kind`, `visibility`, `value`, `display`, `source` and `caveat`. Derived records also carry `derivation` (formula and input record IDs); they are arithmetic on frozen values, such as the margin, the calibration rank and the previous-bin smoothed score. `display` strings are the only number formats the findings use.

## Findings ledger

| ID | Visible | Rule |
|---|---|---|
| F-MARGIN | both | smoothed score over threshold; called marginal below 50% |
| F-CARRYOVER | both | raw score at the alert bin below the threshold; previous-bin smoothed score from the EWMA update |
| F-EMERGE | both | largest one-bin rise in smoothed-contribution percentile: abrupt, already elevated, or gradual |
| F-CF-SPEC | both | single-group model-space repairs that clear the alert |
| F-CF-PERSIST | both | a repair clears the alert through persistence while the smoothed score stays above the threshold |
| F-HYP-TIE | both | candidates stay open and unranked |
| F-SCALE | both | training interquartile ranges of the measured-context features, and where the alert-bin value falls |
| F-GAP | both | missing bins in the past-only window, and whether a gap restarts smoothing |
| F-IDENT | both | leakage and downstream demand cannot be separated without a demand observation |
| F-BRIDGE | both | real-asset physical reasoning and actions are withheld |
| F-LABEL, F-TIMING, F-STATUS, F-BENCH | retro | provider label, timing against onset, case selection, TEST benchmark with its caveats |

## Not loaded yet

- Signal definitions and units: the dataset description has not passed the document gate. Records point to `unavailable:definitions`.
- Raw telemetry and the raw-score series around each alert. These would need the optional reproduction check.
- Any OEM manual, SOP or maintenance record for this asset.

`notebooks/syrus_01_case_evidence_review.ipynb` reviews these files: the ledger across cases, carryover arithmetic, calibration ranks, the past-only window, the repairs and the scale table.
