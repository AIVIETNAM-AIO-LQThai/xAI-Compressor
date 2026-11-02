# Syrus prompt size and context selection

## Why

The first live call (`why_alert` on the 2020-05-29 case) was refused with HTTP 429 by the provider's tokens-per-minute limit:

| | Tokens |
|---|---|
| Local estimate of the input (whole case ledger, system prompt, schema) | 8,162 |
| Plus the maximum output | 9,062 |
| Provider limit, tokens per minute | 8,000 |
| Remaining at rejection | 2,363 |
| `retry-after` | 37 s |

Every question sent the whole ledger (65 records, 15 findings, 7 tests, the full response schema) however simple it was. The correction below sends what the question needs and refuses to send anything that does not fit.

## Measuring

`python -m scripts.syrus_tokens [--parts] [--markdown] [--all-cases]` prints, for every acceptance question, the estimated input tokens before (the whole ledger, rendered as before) and after (the selected slice), what the slice leaves out, and whether the question's essential evidence and the findings its purposes require are present. It makes no model call and needs no key.

`backend/syrus_app/tokens.py` holds the estimate: one token per 3 ASCII characters and per 1.5 other characters (Vietnamese costs more), for the messages and the schema. It is pessimistic on purpose; no tokenizer for the fixed model is installed. The first approved live call reports the true figure: with `SYRUS_DIAGNOSTICS=1` each response carries `estimated_in` next to the provider's `tokens_in`, which is how `request_share` should be calibrated.

Parts are measured separately: system instructions, scope header, records (with their notes), findings, tests and pair limits, conversation history, the user request, the JSON schema, and the JSON quoting overhead. The baseline reproduces the live measurement (8,164 against 8,162 reported).

## The budget

`configs/syrus_service.yaml`: `tokens_per_minute: 8000` and `request_share: 0.75`. One request (estimated input plus the 900-token maximum output) may use at most 6,000 tokens, so the estimated input cap is 5,100. The remaining 2,000 tokens are room for estimate error and for another request in the same minute. A retry is sent only if its larger prompt (the failed reply and the correction appended) is still under the cap.

## Selection (`selection.py`, tables in `policy.py`)

The question is matched, in English and Vietnamese, to topics (`FOCUS_PATTERNS`): why the alert fired, how the numbers are computed, how the evidence developed over the window, candidate explanations, repairs, the documented incident, signal contributions, training ranges, benchmark results, energy and simulation, sources. A question that matches nothing keeps the previous turn's topics; with none, it gets the alert explanation. The question picks topics only; nothing is scored or ranked.

For the topics chosen the view contains:

1. **Always**: the alert time, threshold, smoothed score and margin; the four `unavailable:*` records; `bridge.status` (so that any refusal can cite what it must); the data-gap finding `F-GAP`.
2. **The topics' findings**, each with every record it depends on (its inputs).
3. **Every finding the allowed purposes require** at the depth that applies (the user's choice, or the deepest default among the allowed purposes). `what_happened` is allowed only for the alert and timing topics, `possible_causes` only for candidate explanations, `method` only for how-computed questions; `evidence_and_sources` and `outside_evidence` always.
4. **Lean findings**: a finding present without the question being about it comes with its text (which states every figure it carries) and one anchor record, not all its inputs: `F-EVIDENCE-TIMING` with `temporal.window`, `F-CF-NONUNIQUE` with `alert.threshold`. Asking about timing or repairs loads the full records.
5. **Candidate tests** only when `possible_causes` is allowed, as: which candidates each test separates and whether it needs an external or dataset observation. The follow-up question text is still rendered by the server from the ledger.

A `SCOPE` line tells the model what is and is not loaded (system rule 14), and the schema is built from the selected IDs only (records, findings, tests, purposes), so the model cannot cite anything outside the view.

### Validation keeps the whole ledger

The validator, the required-finding check ("Also relevant" lines come from the full finding text), the qualification triggers and the follow-up question all use the complete case ledger. In addition `check` rejects any reply that uses an ID or purpose this turn did not offer (`not_offered`), so a model that cites outside its view fails closed even if the provider does not enforce the schema's enums.

### Fail closed

- **`context_incomplete`**: a finding a topic or purpose needs, a record a finding depends on, or a core record is missing from the ledger. Nothing is sent.
- **`context_over_budget`**: the prompt is still over the cap after the steps below. Nothing is sent.
- Steps taken to fit, in order, each deterministic and logged: earlier exchanges dropped (they are not evidence), the candidate-test table dropped (it serves only the optional follow-up question), then the lowest-priority matched topic dropped **whole**. A topic is never cut partway, and a finding the allowed purposes require is never removed: dropping a topic also removes the purposes only it allowed. When topics were dropped, the model is told in its context and the server adds a bilingual note to the answer (`narrowed`) naming them.

Offline reasons returned for these are `context_incomplete` and `context_over_budget`; neither calls the model.

## Before and after, acceptance questions (estimated input tokens)

Case 2020-05-29 unless the fixture names another. "After" parts are system / scope / records / findings / tests / history / request / schema / JSON overhead. The largest result on any of the four cases is 4,558 (`persistence_path_repair`), against the cap of 5,100.

| Question | Turn | Before | After | Topics | After, by part |
|---|---|---|---|---|---|
| why_alert | 1 | 8,164 | 3,305 | alert | 1034/62/1307/143/1/0/24/661/73 |
| definitely_a_leak | 1 | 8,169 | 3,942 | attribution | 1034/64/1147/443/324/0/30/793/107 |
| definitely_a_leak_vi | 1 | 8,174 | 3,947 | attribution | 1034/64/1147/443/324/0/34/793/108 |
| dv_pressure_cause | 1 | 8,167 | 4,299 | attribution, signals | 1034/67/1440/443/324/0/27/841/123 |
| motor_current_fix | 1 | 8,180 | 3,868 | repair, signals | 1034/65/1751/193/1/0/40/673/111 |
| predicted_early | 1 | 8,169 | 2,818 | label | 1034/62/781/252/1/0/29/603/56 |
| energy_wasted | 1 | 8,166 | 2,463 | energy | 1034/62/601/116/1/0/27/572/50 |
| apply_simulated_schedule | 1 | 8,172 | 2,469 | energy | 1034/62/601/116/1/0/32/572/51 |
| raw_trace | 1 | 8,165 | 2,378 | sources | 1034/62/601/36/1/0/25/568/51 |
| maintenance_log_vi | 1 | 8,172 | 2,385 | sources | 1034/62/601/36/1/0/32/568/51 |
| oem_manual | 1 | 8,172 | 2,385 | sources | 1034/62/601/36/1/0/32/568/51 |
| signal_definition | 1 | 8,163 | 2,736 | signals, sources | 1034/65/894/36/1/0/23/616/67 |
| air_dryer_2020 | 1 | 8,171 | 4,081 | attribution, label | 1034/66/1201/513/324/0/31/803/109 |
| explain_prev_smoothed | 1 | 8,167 | 3,124 | method | 1034/62/1207/112/1/0/27/623/58 |
| persistence_path_repair (2020-07-15) | 1 | 8,213 | 4,558 | alert, repair, signals | 1034/67/2030/543/1/0/48/717/118 |
| operator_statement | 1 | 8,183 | 3,956 | attribution | 1034/64/1147/443/324/0/43/793/108 |
| ignore_rules | 1 | 8,170 | 3,943 | attribution | 1034/64/1147/443/324/0/31/793/107 |
| two_turn_follow_up | 1 | 8,164 | 3,305 | alert | 1034/62/1307/143/1/0/24/661/73 |
| two_turn_follow_up | 2 | 8,471 | 4,137 | timing | 1034/62/1507/354/1/305/27/702/145 |

The second turn's "before" and history use a synthetic earlier answer of the size three quick units allow. Every row keeps all of the question's essential evidence and every finding its allowed purposes require (`tests/test_syrus_context.py` checks this on all four cases).

The system prompt grew from 969 to 1,034 tokens (rule 14); the schema shrank from 1,121 to 570-840 because its enums list only the offered IDs.

## What the model no longer sees, and what changed

Nothing is removed from the ledger, the case files, the validator, the required-finding check, the qualifications or the follow-up questions. What the model sees for a question is smaller:

- **Findings and records outside the question's topics.** For a plain alert explanation the model no longer sees the candidates, the counterfactual repairs, the per-bin evidence history, the documented incident, the benchmark statistics, the training-range features or the test table; it sees the margin, the smoothing memory, the signal shares, the detector rules and the standing records. It therefore cannot name a candidate or cite a repair in that answer (the validator would have rejected it), and the question's topic decides which of the model's earlier freedoms remain. A question that spans topics the budget cannot hold together is narrowed, with the note described above.
- **Per-bin and per-repair records for lean findings.** `F-EVIDENCE-TIMING` and `F-CF-NONUNIQUE` keep their full text and figures; the thirteen `temporal.*` records and the six `cf.*` records are loaded only when the question is about timing or repairs.
- **Candidate-test wording.** The model sees which candidates a test separates and whether it needs an external or dataset observation, not the free-text description; maturity and result appear once when they are uniform ("all draft, result not_run") and per test otherwise. The follow-up question the user sees is unchanged.
- **Purposes.** The schema lets a turn declare only the purposes its topics allow, which also fixes which findings an answer must cover.
- **History.** Earlier answers are replayed as the model's claim texts only; the qualifications, "Also relevant" lines, operator note and follow-up question the server wrote are re-derived each turn and not re-sent.
- **Wording.** System rule 14 and the `SCOPE` line are new.

No qualification, refusal rule, depth budget, spending cap or the model changed.

## Diagnostics for development

`SYRUS_DIAGNOSTICS=1` adds a `diagnostics` object to turn responses, offline responses and `failed_checks`: the topics, dropped topics, depth, history sent, estimated tokens by part and in total, the cap, and per call the estimated and actual tokens plus provider fields. Provider fields are limited to: HTTP status, `retry-after`, the token and request limit, remaining and reset headers, the request ID, the error type and code, the model name the provider reports, and for a rate-limit error the three integers in its message (limit, used, requested). Each is checked against a strict pattern before it is kept. Prompts, answers, the provider's error text, its echoed output and organisation names are never read into diagnostics. The same object is logged to `syrus.provider` (failures) and the turn log carries topics and the estimate; neither carries message text. `scripts/syrus_smoke.py` turns diagnostics on and prints them.

## Not yet verified

- The estimate against the provider's real prompt token count (the first approved live call settles it; adjust `request_share`).
- That strict structured output accepts a schema whose enums differ from question to question (by topics and allowed purposes), and whether the provider's per-schema preparation adds latency. The single-value `["none"]` enum is used where a list must stay empty.
- Whether a retry in the same minute as a heavy first call is refused: the retry rule above approximates it, and a local per-minute governor was not added.
- The topic patterns are keyword lists for English and Vietnamese. A phrasing they do not match falls back to the previous topics or the alert explanation; the Vietnamese patterns and topic names need the owner's review.
