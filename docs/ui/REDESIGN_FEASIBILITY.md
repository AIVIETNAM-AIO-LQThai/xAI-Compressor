# AeroXAI redesign: feasibility and design review

Status: decisions approved on 2026-11-03 as recommended (units: charts say "units not loaded" until the dataset description passes the document gate). Stages 1 to 3 implemented (Syrus fixes; explorer index and API; shell, explicit selection, Syrus drawer and Overview). Every number below was checked read-only on 2026-11-03 against the local files.

## Summary

- **The whole recorded history is browsable with exact provenance.** The raw MetroPT-3 CSV is local (1,516,948 rows at 10 s, 2020-02-01 to 2020-09-01, zip checksum matches `configs/metropt.yaml`). It reproduces the frozen five-minute features exactly (max difference 0.0 on the bins checked). Refitting the frozen PCA detector from `train.parquet` gives the frozen threshold (52,941.88), the frozen Incident 2 smoothed score (58,988.95) and the 142 test alert episodes, in 0.3 s. A real data explorer needs an index and an API, not new data.
- **The digital twin cannot feed the frozen detector.** It is a generic 10 m³ receiver with a three-compressor fleet at 7 bar. It produces pressure, flows and power for a different plant, and none of the 15 MetroPT-3 channels. Scoring it with the frozen PCA would be invented, so the twin is labelled "detector not applicable".
- **What the frozen detector can defensibly score is a declared sensor fault applied to recorded 10 s telemetry,** passed through the unchanged feature pipeline. Leak, demand, regulation and performance scenarios cannot be generated credibly for the MetroPT-3 channels today.
- **The Vietnamese `kind_mismatch` has a confirmed, language-specific cause beyond segmentation:** in Vietnamese, stating the provider label ("rò rỉ khí") matches the Vietnamese name of the leakage candidate, so a label claim cannot pass both the entity check and the kind check. Details in section 6.

## 1. What real data can be browsed now

| Asset | Content | Status |
|---|---|---|
| `data/raw/MetroPT3(AirCompressor).csv` | 15 channels at 10 s, 1,516,948 rows; 331 gaps over 60 s, longest 48 h | Local, checksum-verified through the zip |
| `data/processed/features.parquet` | 50,301 valid five-minute bins (coverage ≥ 0.80), 63 features + `sample_count`, `coverage_ratio`; 312 gaps over 5 min | Local; equals the raw-data pipeline output |
| `train` / `calibration` / `test` parquet | 5,271 / 1,796 / 43,221 bins, the frozen chronological splits | Local |
| Frozen PCA detector | Not stored as a file; refit deterministically (`svd_solver="full"`) from `train.parquet` | Reproduces the frozen threshold and episodes exactly |
| TCN / Transformer scores | `data/processed/*_scores.parquet` | Experimental; not production, not shown as the detector |

**Indexing needed (one reproducible build script, outputs gitignored under `data/processed/explorer/`):**

1. Raw telemetry split by day into parquet (read time for any day well under a second).
2. The detector series for all 50,301 bins: raw score, smoothed score, threshold hit, alert, EWMA restarts after gaps, split membership.
3. The 142 test episodes with their incident linkage under the existing evaluation rule (9 linked, 133 not linked).
4. A manifest with source checksums and verification checks: the build fails unless it reproduces the frozen threshold, the Incident 2 smoothed score and the 142 episodes.

**Rules the explorer keeps:**

- Gaps stay gaps: no line is drawn across a missing bin, and every EWMA restart is marked.
- Training-period scores are in-sample and calibration scores set the threshold. Both periods are shaded and labelled; only the test period is the evaluation.
- Long ranges are downsampled with min/max per bucket so no spike or alert disappears.
- Unlabelled time is never coloured "healthy"; an alert episode not linked to a documented incident is "not linked", never "false alarm".
- **Units:** the dataset description PDF has not passed the document gate, so charts would say "units not loaded". **Decision for you.**

## 2. What the digital twin can generate

`ml/twin/physics.py` integrates `dp/dt = (R·T/V)(m_in − m_demand − m_leak)` for an isothermal 10 m³ receiver. `configs/compressors.yaml` defines two fixed compressors and one variable-speed compressor, a 6.8–7.2 bar band, and a proportional controller (`ml/control/baseline.py`). `/twin/simulate` takes constant flows for at most an hour.

**It can generate (labelled SIMULATED, generic plant):**
- receiver pressure
- the inflow, demand and leak mass flows given as inputs
- compressor on/off and load, and power, from the fleet model

**Scenarios it supports:**
- **Baseline cycles:** with the existing controller.
- **Demand change and leak:** both are additive outflows.
- **Regulation change:** a different pressure band or setpoint.

**Leak versus demand:** the twin shows they are indistinguishable. Equal total outflow gives identical pressure, which is the honest demonstration.

**It cannot generate:**
- TP2, TP3, H1, DV pressure, reservoirs, oil temperature or motor current
- any of the 8 digital MetroPT-3 signals
- sensor faults
- compressor degradation

**New simulation code needed for the lab:** a time-varying schedule (onset, severity, duration), the controller loop, and a seed for demand noise. This would be a new endpoint, leaving `/twin/simulate` and its frozen results untouched.

## 3. Can the frozen PCA score synthetic data?

**Twin output: no.** The detector needs all 15 channels at 10 s to build 63 features. Inventing the other 14 channels, the digital state sequences, oil temperature and motor current from a pressure curve would be fabrication. The pressure scale (7 bar plant against MetroPT-3 TP3 around 9 bar) would also sit outside training support. The lab shows "Detector: not applicable to this model".

**Declared perturbations of recorded telemetry: yes, for sensor faults.** Take a recorded 10 s background window from the test period, apply a declared change, then run the unchanged pipeline. That is schema normalisation, five-minute features, the frozen scaler and PCA, EWMA 0.2 with restarts after gaps, the frozen threshold, and the 3-of-4 rule.

| Scenario | Applied to raw samples | Defensible? |
|---|---|---|
| Sensor offset (step) | constant added to one analog channel from onset | Yes, exact |
| Sensor drift | linear ramp on one analog channel | Yes, exact at 10 s |
| Flatline / stuck | channel held at its value at onset | Yes |
| Added noise | seeded Gaussian noise on one channel | Yes |
| Dropout | samples removed, so bins fall below 80% coverage and leave the model's view | Yes, an exact pipeline effect |
| Digital stuck-at | one digital signal held at 0 or 1 | Yes, as a sensor fault |
| Leak, demand, regulation, compressor performance | would need consistent changes to pressure cycles, compressor duty, motor current and oil temperature | **No**: no validated MetroPT-3 physics model; point users to the four recorded air-leak incidents instead |

**What every run shows:**
- the known injected change, as SIMULATED ground truth
- the detector's raw and smoothed score, threshold and alert state, as the detector's response
- whether any feature left its training range (the existing training-range check), as a visible support warning
- a plain "no alert" when the detector does not respond

The detector does not identify the injected fault, and the lab says so.

## 4. Choosing what to inspect

**Today:**
- The Overview's Syrus button is hard-wired to `investigation/c-h3vz/retro`.
- `Investigation.tsx` falls back to `cases[1]` when no case is given.
- `syrus_cases.yaml` marks Incident 2 `featured`.

**Proposed selection model:**

| Selection | Deep link | Investigation content | Syrus |
|---|---|---|---|
| Documented incident (4) | `#investigate/incident/<case>/<asof\|retro>` | Reviewed case file | Enabled (existing contract) |
| Alert episode not linked to an incident (133) | `#investigate/episode/<start>` | Computed with the frozen detector, labelled "computed, not reviewed" | Off: scope note |
| Recorded interval | `#explore/<from>/<to>` | Charts, scores, alerts, coverage | Off: scope note |
| Synthetic run | `#lab/<scenario>/<seed>` | Run parameters, injected change, detector response | Deterministic simulation notes only |

- **No default selection.** Investigation without a selection shows a picker. "Guided example: Incident 2" is one labelled choice among others, and `featured` is no longer read by the interface.
- **The URL is the source of truth;** the last selection is also kept for the browser session so navigating away and back returns to it.
- **Syrus context follows the selection's identifier:** `case:<id>:<mode>`, `episode:<start>`, `interval:<from>/<to>`, `sim:<hash>`. Each context has its own conversation, and context types without a reviewed contract never receive another case's findings.
- **Later, if wanted:** reviewed case files for selected unlinked episodes. This needs a new verification artifact, built and reviewed the same way as the four incident files, never by reusing them.

## 5. Layouts

**Desktop shell:** left navigation (Overview, Explorer, Investigation, Scenario Lab, Evidence Proof). A selection bar under the page title shows what is selected and in which mode. A static Syrus launcher sits bottom-right: the existing mark as a fixed round button with a context label. It opens a 440 px drawer over the right edge. There is no permanent Syrus column and no motion.

```
OVERVIEW
+--------------------------------------------------------------------------------+
| AeroXAI   Recorded data: MetroPT-3 APU, 1 Feb – 1 Sep 2020, 10 s, 15 channels    |
| [Valid 5-min bins 50,301 of 61,392 · 81.9 %] [Alert episodes 142 · test] [4/4 timely] |
| [Not linked to incidents 133 · not "false alarms"]       [Mode: Advisory]       |
+--------------------------------------------------------------------------------+
| Alert activity, Feb–Sep 2020 (smoothed score, threshold, episodes, gaps)        |
| ▁▁▂▁▃▁▁▂▁▁▇▁▁▂▁▁▃▁▁█▁▁▁▂▁▁▁▃▁▁▁▅▁▁   train | calib |            test               |
+---------------------------------------+----------------------------------------+
| Start here                            | Documented incidents (retrospective)   |
| [Explore recorded data]               | 1 · 18 Apr  2 · 29 May  3 · 5 Jun  4 · 15 Jul |
| [Browse alert episodes]               | Guided example: Incident 2             |
| [Open Scenario Lab · SIMULATED]       |                                        |
+---------------------------------------+----------------------------------------+
```

```
REAL DATA EXPLORER
+--------------------------------------------------------------------------------+
| Range [29 May 2020 18:00] – [30 May 06:00]  [1h][6h][24h][7d]  Episodes ▾  Labels ☐ |
+--------------------------------------------------------------------------------+
| Smoothed score ——  threshold - - -  hits ▪  alert ███  EWMA restart |  gap ░░  |
+--------------------------------------------------------------------------------+
| TP2 / TP3 / reservoirs (10 s ≤ 24 h, else 5-min mean with min–max band)         |
| Oil temperature · Motor current                                                |
| Digital states: COMP DV LPS MPG towers … (on/off strips)                        |
+--------------------------------------------------------------------------------+
| Coverage per bin ▁▁▁▁▁▁▁                                                       |
| Selected bin 21:20 → raw 35.8 · smoothed 58,989 · alert ON → [Investigate]     |
+--------------------------------------------------------------------------------+
```

```
INVESTIGATION (selected incident or episode)
+--------------------------------------------------------------------------------+
| Incident 2 · 29 May 2020 21:20 · Retrospective [Past-only|Retrospective]  Change |
| [58,989] [52,942] [+11.4%] [rule 3 of 4]                                         |
+---------------------------------------+----------------------------------------+
| Group contributions (bars)             | Evidence over the window (lines)       |
+---------------------------------------+----------------------------------------+
| Counterfactual repairs (bars)          | Candidates · tests · missing evidence  |
+---------------------------------------+----------------------------------------+
| Verified findings (3) · View all       | Sources                          [Syrus]|
+--------------------------------------------------------------------------------+
```

```
SCENARIO LAB                                                         SIMULATED
+------------------------------+-------------------------------------------------+
| Model [Generic receiver twin | Recorded-background sensor fault]                |
| Scenario ▾  Onset ▭ Severity ▭ | Pressure / signal (injected change marked)     |
| Duration ▭  Seed ▭            | Flows or affected channel                        |
| Background window ▾ (recorded)| Detector response: score · threshold · alert   |
| [Run] [Pause] [Replay] [Reset]| — or "Detector not applicable to this model"    |
| Support check: 2 features out of training range ⚠                               |
| Ground truth (injected): offset +0.4 on TP3 from 12:00 · seed 17                |
+------------------------------+-------------------------------------------------+
```

```
MOBILE (375 px)
[≡ AeroXAI]                [Syrus ◉]
Selection: Incident 2 · Retro  ▾
[58,989] [52,942] [+11.4%]
Score timeline (swipe to pan)
Tabs: Signals | Evidence | Findings
Syrus opens full screen; context chip at top.
```

**Where every major chart comes from:**

| Chart | Source |
|---|---|
| Overview KPIs | Explorer index (bins, coverage, episodes); `docs/detector_benchmark.json` for 4/4, 2/4 and 0.886 |
| Alert activity strip, explorer score timeline | Detector series rebuilt from `features.parquet` with the frozen detector, checked against frozen values |
| Documented incident bands | `configs/metropt.yaml`, shown as retrospective labels and hidden in past-only views |
| Sensor traces and digital strips | Raw CSV via the daily index (10 s); five-minute aggregates for long ranges |
| Coverage | `coverage_ratio` in `features.parquet` |
| Investigation (four incidents) | Reviewed case files (unchanged) |
| Investigation (episodes) | `ml/explainability/pca.py` contributions and the temporal evidence, computed and labelled as such |
| Scenario Lab: generic twin | `ml/twin` + `compressors.yaml` + baseline controller (SIMULATED) |
| Scenario Lab: sensor faults | Recorded raw background + declared perturbation (SIMULATED change) → unchanged pipeline → frozen detector |

## Decisions that would look impressive but mislead (not recommended)

- **Health %, uptime, remaining useful life, or "probability of leak".** No model supports any of them.
- **"Predicted failure" for an alert before onset.** The detector raises anomaly alerts; the temporal models are experimental.
- **Leak or demand scenarios scored by the frozen detector** via invented channel mappings.
- **Colouring unlabelled periods green** or calling unlinked episodes false alarms.
- **Lines interpolated across gaps,** or a smooth "trend" line drawn through downsampled data that hides spikes.
- **Counting synthetic runs** into the benchmark KPIs.
- **Replay animated as "live".** Replay of recorded or simulated values is labelled RECORDED REPLAY or SIMULATED.
- **Showing training-period scores as evaluation results.**

## 6. The Vietnamese failure and `busy`, investigated offline

**The question:** "Có nhật ký bảo trì hoặc tài liệu kỹ thuật nào xác nhận sự cố này chắc chắn do rò rỉ khí không? Nếu chưa có, cần bổ sung bằng chứng gì?"

It selects `certainty`, `label` and `sources`. It requires F-LABEL and F-IDENT, and offers records of seven kinds.

**First mechanism.** Answering both parts in one sentence makes a unit cite the label (`dataset_label`) together with the candidates (`hypothesis`). The validator returns `kind_mismatch`, which is correct. The correction message names the failed check but does not say how to fix it ("split it, one unit per kind").

**Second mechanism.** This is a validator conflict specific to Vietnamese. Even a correctly split label unit fails with `entity_not_cited: candidate air_leakage`. The Vietnamese label wording "rò rỉ khí" matches the Vietnamese name of the leakage candidate ("Rò rỉ khí"), and the only way to satisfy that check is to cite `hyp.air_leakage`. That then breaks the kind rule for a `dataset_label` unit. In English, "Air leak" does not match the candidate name "Air leakage", so English answers pass. Verified offline: the merged reply fails with both codes, and the split reply still fails `entity_not_cited`.

**Proposed fix, keeping both checks:**
- The label record declares which candidate its wording names. A label unit may then use that name without citing the hypothesis record, and only that one.
- Give targeted correction text for `kind_mismatch` and `entity_not_cited`.
- Add a sanitised diagnostic listing each unit's kind and the kinds it cited (IDs and kinds only, no text), so the next live failure can be confirmed exactly.
- Add English and Vietnamese tests covering:
  - document availability
  - the label
  - identifiability
  - missing external evidence
  - the two-part follow-up

**`busy`, simulated offline with live-like token counts.** The two-call failed turn left 4,360 tokens in the governor's minute window. A re-send 15 s later needed 3,067 more, over the 7,200 allowance, so it was refused for 41 s.

- No provider call was made and the question was not duplicated; the client's "Try again" replaces the failed turn.
- **The message is wrong for one user:** "Syrus is answering someone else". Proposed: separate `busy` (another call in flight) from `paced` (minute budget), each with its own text and countdown.
- Fixing the validator conflict removes the wasted retry that caused it.

## 7. Risks, validation and milestones

**Risks:**
- **Payload size:** a 7-day range at 10 s is 60,000 points per channel. This is handled with resolution switching and min/max downsampling, plus maximum range sizes on the API.
- **Start-up time and memory:** the detector refit and index loading have to stay fast and lazy.
- **Explorer scope confused with Syrus scope:** handled by explicit context chips and disabled states.
- **Units not loaded (document gate).**
- **Twin conclusions read as MetroPT-3 behaviour:** handled by labels plus a visible "generic receiver model".
- **The heavy raw-data endpoints:** these must stay off the public build or be rate-limited.

**Validation:**

| Area | Checks |
|---|---|
| Index | Build-time check against the frozen threshold, the Incident 2 values and the 142 episodes; no bins outside the source; gaps preserved |
| Synthetic runs | Determinism per seed; causality (nothing changes before onset); units unchanged; only the declared channel changes; feature recomputation equals the pipeline; coverage handling; EWMA restarts after gaps; 3-of-4 alert decisions |
| Twin | Its existing analytical checks, plus leak/demand equivalence at equal total outflow |
| Interface | Scripted runs at 1920, 1440, 820 and 375 px |

**Milestones (Sonnet, one at a time, each ending with tests, build and screenshots):**

1. **Syrus fixes:** the label/candidate naming fix, correction texts, the unit-kind diagnostic, splitting `busy` from `paced`; English and Vietnamese tests. Small and independent.
2. **Explorer data service:** index build script with manifest, read-only API (`/explorer/overview`, `/explorer/range`, `/explorer/episodes`, `/explorer/bin`), verification tests.
3. **Shell and selection:**
   - new Overview
   - selection bar and URL model
   - Investigation without a fallback
   - Syrus as a launcher and drawer, with the existing conversation code reused
4. **Real Data Explorer interface:** ECharts is already a dependency.
5. **Investigation for unlinked episodes:** computed and labelled, with Syrus scoped off.
6. **Scenario Lab:** the generic twin with schedules, then sensor faults on recorded background with the frozen detector.
7. **Full QA:** screenshots, before/after comparison, docs.

## Decisions needed before implementation

1. Approve using the local raw CSV for the explorer and sensor-fault lab (no download needed; it is already checksum-verified).
2. Units and signal definitions: approve the dataset description document (G-DOC), or ship charts that say "units not loaded".
3. Scenario Lab scope: the generic twin (detector not applicable) plus sensor faults on recorded background (frozen detector). Leak, demand, regulation and performance are excluded from detector scoring.
4. Unlinked episodes: computed evidence with Syrus off now, and reviewed case files later only if wanted.
5. Order: Syrus fixes first, then the explorer data service.
