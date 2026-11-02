# Syrus case files

Deterministic evidence for Syrus, built from frozen AeroXAI artifacts. Nothing here retrains, retunes or re-thresholds the detector. `LEDGER_REVIEW.md` records the ledger decisions and the response contract; `SERVICE.md` describes the chat service built on these files.

```bash
python -m scripts.build_syrus_cases          # rebuild docs/syrus/cases/
python -m scripts.build_syrus_cases --check  # fail if the committed files differ from a fresh build
```

## Inputs

- Frozen reports: `docs/xai_report.json`, `docs/xai_verification_report.json`, `docs/detector_benchmark.json`, `docs/proof_manifest.json`.
- Configs: `configs/metropt.yaml`, `configs/detection.yaml`, `configs/evidence_action_bridge.yaml`.
- Syrus configs: `configs/syrus_cases.yaml` (opaque case IDs, display floor), `configs/syrus_hypotheses.yaml` (unranked candidates in a process and an evidence layer), `configs/syrus_discriminators.yaml` (diagnostic tests: compared pairs, observations and where they live, maturity, rule, result, follow-up questions). The last two are drafts that need domain review.
- Code pointers (file, symbol, hash) to `ml/detection/alerts.py`, `ml/explainability/verification.py` and notebook 04 for the smoothing, persistence, repair and alert-start rules.

`manifest.json` records the content hash of every input. JSON inputs are hashed as canonical JSON and text inputs with LF line endings, so the hashes do not depend on Git line-ending settings.

## Files

One pair per case, listed in the manifest by alert time:

- `{case_id}.retro.json`: every record, with full source pointers (`file`, `path`, `sha256`). Paths use `[field=value]` selectors, so they do not depend on list order.
- `{case_id}.asof.json`: only records visible at alert time. Source pointers keep the file and hash but drop the path or symbol, which could reveal the incident number. It has no label, onset, timing, case status, benchmark or document content.

Both files also carry `tests` (the diagnostic test table) and `pair_limits` (candidate pairs that cannot be told apart with this data).

Each record has `id`, `kind`, `visibility`, `value`, `display`, `source` and `caveat`. Derived records also carry `derivation` (formula and input record IDs); they are arithmetic or logic on frozen values, such as the margin, the calibration rank, the previous-bin smoothed score, the share carried by smoothing and the hit pattern implied by the persistence rule. `display` strings are the only number formats the findings use.

## Findings ledger

Each finding has a short `text` and, where useful, a technical `detail`. The decisions behind the rules, the candidate and test semantics, and the response contract for the Syrus service are in `LEDGER_REVIEW.md`.

| ID | Visible | Rule |
|---|---|---|
| F-MARGIN | both | smoothed score over threshold, as a number only |
| F-EWMA-MEMORY | both | fires when the alert bin's raw score is below the previous smoothed score; share of the score carried from earlier bins |
| F-EVIDENCE-TIMING | both | per displayed group: percentile at window start and alert, largest one-bin rise, first bin above every calibration bin |
| F-CF-NONUNIQUE | both | fires when two or more tested single-group repairs clear the alert; scope, reduction ratios and pre-window bound in `detail` |
| F-CF-PERSIST | both | fires when a repair clears the alert through persistence while the smoothed score stays above the threshold |
| F-TRAIN-RANGE | both | where each exported measured feature falls against its training interquartile range |
| F-GAP | both | missing bins in the past-only window, and whether a gap restarts smoothing |
| F-CANDIDATES | both | candidates stay open and unranked; pair limits |
| F-IDENT, F-BRIDGE | both | asset-wide: leak and demand not separable without a demand observation; physical bridge withheld |
| F-LABEL, F-TIMING, F-SELECTION | retro | provider label, timing against onset, case selection |
| F-ALERT-BURDEN, F-DISCRIMINATION | retro | timely recall and benchmark-false rate; PR-AUC and alert-bin profiles, with the TEST selection caveat |

## Not loaded yet

- Signal definitions and units: the dataset description has not passed the document gate. Records point to `unavailable:definitions`.
- Raw telemetry and the raw-score series around each alert. These would need the optional reproduction check.
- Any OEM manual, SOP or maintenance record for this asset.

Review notebooks:

- `notebooks/syrus_01_case_evidence_review.ipynb` reviewed the first ledger version (case schema v1); its saved outputs describe that version.
- `notebooks/syrus_02_ledger_revision_review.ipynb` reviews the current files: the smoothing-memory criterion, the hit patterns implied by the persistence rule, repair paths and reduction ratios, the pre-window bound, evidence timing, the display floor, and the candidate and test tables. Tables are also saved to `outputs/syrus_02/`.
