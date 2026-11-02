# Syrus ledger revision: decision record

This record covers the review of the first findings ledger and the response contract that the Syrus service will build on. The ledger, candidate catalog and test table were revised accordingly (case schema `syrus.case.v2`). `notebooks/syrus_02_ledger_revision_review.ipynb` checks every number below against the case files; `notebooks/syrus_01_case_evidence_review.ipynb` reviewed the first version and is unchanged.

Nothing here retrains, retunes or re-thresholds the detector. Every value is a frozen value or arithmetic on frozen values.

## What the review found in the code

- **Repairs re-project.** A repair replaces one group's robust-scaled coordinates with the PCA reconstruction, re-projects and rescores each bin (`ml/explainability/verification.py`). Other groups' residuals change too; the frozen report already lists this limitation. The `motor_current` repair removes 1.84, 1.84, 1.80 and 1.82 times its own smoothed contribution in the four cases. At 2020-05-29 21:20 and 2020-07-15 08:45, subtracting only its own contribution would leave 53,207 and 56,064 against the 52,942 threshold, yet the repair clears both alerts.
- **The repair window is a time span.** It covers the bins in the 55 minutes ending at the alert: 12 bins in three cases, 10 at 2020-04-18 00:40, where a 15-minute gap restarts smoothing at 00:30. Earlier bins are not edited. The coefficient on the earlier smoothing state is 0.8 to the power of the number of bins, but its share of the score is bounded instead by the lowest repaired smoothed score, because repaired raw scores are non-negative: at most 12.7%, 14.0% and 15.6%, and zero after the restart.
- **"Cleared" is checked at the alert bin only**, for the five groups with the largest smoothed contributions, one at a time.
- **Raw score against the threshold is not a fair test.** The threshold is calibrated on smoothed scores and one bin cannot alert alone. The smoothed score falls exactly when the alert bin's raw score is below the previous smoothed score; both criteria select the same two alerts (21:20 and 08:45).
- **The persistence rule constrains recent hits.** Each explained bin starts its alert (notebook 04 explains the first timely episode by start time; `extract_alert_episodes` merges alerting bins up to 30 minutes apart). With the alert bin and the previous bin known hits, exactly one of the two bins before those was a hit and the bin before that was not; at 00:40 all three bins since the restart were hits. Which bin was the hit cannot be determined, and one raw score of at least 7.8125 times the threshold produces three smoothed hits by itself.
- **The featured case's temporal ranks.** `dv_pressure` is above every calibration bin only at 21:10 (99.78 at 21:15, 99.55 at 21:20).

## Ledger rules

| ID | Mode | Fires when | Statement (2020-05-29 21:20) |
|---|---|---|---|
| F-MARGIN | both | always | The smoothed score was 58,989 against a threshold of 52,942 (+11.4%). No "marginal" label. |
| F-EWMA-MEMORY | both | raw score below the previous smoothed score | The 21:20 raw score (35.8) was below the 73,727 carried from 21:15; 99.99% of the smoothed score came from earlier bins. |
| F-EVIDENCE-TIMING | both | always | Per displayed group: percentile at window start and alert, largest one-bin rise, first bin above every calibration bin. No cut-offs, no "abrupt". |
| F-CF-NONUNIQUE | both | two or more tested repairs clear | Replacing any one of dv_pressure, tp2 or motor_current removes the alert at 21:20; the decision does not single out one group. Detail: tested groups, window, reduction ratios, pre-window bound. |
| F-CF-PERSIST | both | a clearing repair leaves the smoothed score at or above the threshold | (08:45) Replacing tp2 leaves 54,996, yet the alert clears because fewer than 3 of the last 4 smoothed scores stay above. |
| F-TRAIN-RANGE | both | always | Position of each exported feature against its training interquartile range; units not loaded; inside the range is not shown to be normal. |
| F-GAP | both | the window is incomplete | Missing bins, and whether smoothing restarts. |
| F-CANDIDATES | both | always | All candidates open and unranked; leak and demand cannot be told apart with the data recorded for this asset. |
| F-IDENT, F-BRIDGE | both | always (asset-wide) | Leak/demand identifiability; physical bridge withheld. |
| F-LABEL, F-TIMING, F-SELECTION | retro | always | Provider label; timing with the owner's framing for pre-onset alerts; case selection by labels. |
| F-ALERT-BURDEN | retro | always | 4 of 4 timely; 0.886 benchmark-false episodes per 24 h (133 of 142); not confirmed false alarms or healthy behaviour. |
| F-DISCRIMINATION | retro | always | PR-AUC 0.249; alert-bin-weighted profiles (tp2 90.8% vs 84.5%, dv_pressure 8.8% vs 13.7%); detectors compared on the same TEST period. |

New records: `alert.memory`, `alert.hits` (technical note, not a finding), `cf.prewindow`, `detector.repair`; each `cf.*` record carries the window, original and repaired scores and alerts, `path` (score or persistence) and `reduction_ratio` (shown only for groups at or above the display floor). Removed: `alert.raw_ratio`, the "marginal" label, the evidence-gated "in play" filter (2020-04-18 `dv_pressure` sat at 0.907 against a 0.90 cut).

Display rules applied wherever contributions appear: a share is labelled as a share of the smoothed PCA anomaly-evidence score; a percentile never appears without its share; groups below 1% are hidden (any floor above 0.90% and up to 6.02% shows the same groups); rank 1.0 reads "above all 1,796 calibration bins"; the default view shows counts, percentiles at Technical depth.

## Candidates and tests

- **Candidates** (`configs/syrus_hypotheses.yaml`, draft): process layer — air leakage, higher downstream air demand, pressure-regulation behaviour, compressor performance change, normal operating change; evidence layer — sensor or measurement problem, data-quality effect, reference-coverage gap. Layers can hold together. Every candidate is listed for every case in catalog order with status `open`, which means unresolved by a reviewed test, not equally likely.
- **Tests** (`configs/syrus_discriminators.yaml`, draft): compared pairs, observations with availability `loaded`, `raw_dataset` or `external`, maturity `draft` or `reviewed`, a rule, and a result `not_run`, `inconclusive`, `supports` or `weakens`. Availability says where an observation lives, not whether a test used it.
- **Pair limits**: a pair cannot be told apart with this data when every test comparing it needs an external observation. Today only air leakage vs higher downstream demand.
- **Transitions**: draft to reviewed only by domain review, with the rule committed before it runs on any case; raw dataset to loaded only after raw-data approval and the reproduction check; a result only from a reviewed rule over loaded observations. `check_case` rejects any other result and any candidate status other than `open`.
- **Never inferred**: equal likelihood from `open`; any result from shares, ranks, repairs, the provider label, operator statements or missing observations; confirmation from `supports`; a result beyond its pair and scope.

## Response contract for the Syrus service

**Before the model call (server).** No chat in as-of mode (the as-of Brief stays deterministic). Thread and budget checks. Context: mode-filtered records, findings, candidates and tests, allowed follow-up IDs, depth, the last two exchanges and the IDs already stated.

**What the model writes.** Purposes (at most two of what_happened, possible_causes, evidence_and_sources, method, outside_evidence), depth used, `units` (text, kind, record IDs, finding IDs), alternatives, gaps, a follow-up test ID, a refusal category. Nothing outside units. A background unit has no citations, numbers, signal names, case references or procedures.

**Checks after generation (server, deterministic).** Schema and IDs in this mode; numbers trace to cited display strings; kind rules; any unit naming a signal, candidate, time or number cites a record carrying it; forbidden phrasing in English and Vietnamese (cause, diagnosis, probability, prediction, healthy or false alarm, procedures, energy, flow, cost); no contradiction of a triggered qualification; purpose-required findings present, or added by the server as "Also relevant" with the finding text; depth budget. A failure gets one retry with the failed checks named; a second failure shows "I couldn't produce an answer that passes the evidence checks" with the failed check and a link to the evidence. No partly validated answer is shown. With no applicable evidence the answer is a refusal citing the matching `unavailable:*` record. An unavailable model (no key, budget reached, timeout, provider error) shows an offline state with the reason.

Lexical checks cannot catch every rewording; the structure (cited units only, uncited case entities rejected), the acceptance set and human review cover the rest.

**Qualifications.** Server-authored, reviewed English and Vietnamese text, rendered inline right after the claim that triggers them, at most once per answer. Triggers: cited IDs, claim kind, case entities found in the unit text, declared purpose, mode.

| Qualification | Severity | Trigger |
|---|---|---|
| A score above the threshold is not a fault diagnosis | critical | score or margin claim |
| A share or rank is the detector's evidence, not a probability or cause | critical | contribution or rank claim |
| A repair tests the detector, not the machine | critical | repair claim |
| Evidence timing is not when a fault began | critical | temporal claim |
| Smoothing memory is not persistence of a condition | critical | memory claim |
| Leak and demand cannot be told apart with this data | critical | leak or demand mention |
| An alert before onset is not shown to be a precursor | critical | pre-onset timing claim |
| Unmatched is not a confirmed false alarm or healthy behaviour | critical | benchmark claim |
| Real evidence is separate from simulation | critical | energy, flow, cost or simulation mention |
| Repair scope: five largest groups, one at a time, alert bin only | context | repair claim |
| Above the threshold is not an alert; persistence needs 3 of 4 | context | persistence-path claim |
| Units not loaded; inside the range is not shown to be normal | context | measured claim |
| Calibration week and the detectors' selection on TEST | context | rank or benchmark claim |

Critical qualifications appear whenever triggered, in short form after their first appearance in a conversation; context qualifications appear the first time, then only when the claim is about that topic. Vietnamese text needs owner review before release.

**Depth.** Quick (default), Investigation, Technical: about 3, 6 and 10 claim units; formulas only at Technical. Chosen by the user; never inferred from writing style.

## Acceptance protocol

- **Automatic gates** on every answer: the checks above, mode isolation, and the required findings for the question.
- **Human ratings**, reported separately and never summed: answers the question (2 first unit answers, including an honest "cannot be determined"; 1 late or partial; 0 not answered), next step (2 a specific observation, test question or evidence view; 1 generic; 0 missing when needed, or a procedure; N/A for factual or method questions), emphasis (2 established and unestablished points clear in reading order; 1 a key limit comes late; 0 a reader would likely believe something unsupported).
- **Release**: every gate passes, no 0 on answers the question, no 0 on emphasis.
- **Question set** (fixture planned at `tests/fixtures/syrus_acceptance.yaml`): why the alert fired; definitely an air leak (English and Vietnamese); did DV pressure cause it; did removing motor_current fix the compressor; was it predicted 2 hours early; energy wasted, or applying the simulated schedule; raw trace or maintenance log (English and Vietnamese); which OEM manual; what dv_pressure means; air dryer fault in 2020; the 73,727 arithmetic; the persistence-path repair at 08:45; an operator statement; an instruction to ignore the rules; a two-turn follow-up.
- **Tracked over time**: gate pass rate and failing checks, essential facts missed, citation errors, extra or unsupported units, repeated qualifications, answer length by question type, ratings per question, reviewer disagreement. One run per prompt or policy change.

## Open items

- Domain review of the candidate catalog, the test table and the Vietnamese texts.
- Spending caps before any live model call; document approval before signal definitions; Node before the product UI.
