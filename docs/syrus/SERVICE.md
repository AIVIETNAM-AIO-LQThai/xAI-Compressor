# Syrus service

`backend/syrus_app/` is a slim FastAPI app (requirements in `requirements-syrus.txt`) that serves the case files, mints guest sessions and answers retrospective questions. It imports nothing from `ml/`, `backend/app/` or the scientific stack, and exposes no equipment, simulation or optimisation route. The response contract it implements is in `LEDGER_REVIEW.md`.

## Run

```bash
uvicorn backend.syrus_app.app:app
```

Environment variables, all read on the server only:

| Variable | Effect |
|---|---|
| `SYRUS_LLM_API_KEY` | Groq key. Absent: Syrus reports `key_missing` and makes no call. |
| `SYRUS_CAPS_CONFIRMED=1` | Set only after the owner confirms the spending caps in `configs/syrus_service.yaml`. Absent: `caps_unconfirmed`, no call. |
| `SYRUS_SESSION_SECRET` | Signs guest tokens. Absent: random per process, so sessions end on restart. |
| `SYRUS_IP_SALT` | Salts the IP hash kept in turn metadata. |
| `SYRUS_BUDGET_FILE` | Spend file (default `outputs/syrus_budget.json`, gitignored). |
| `SYRUS_KILL_SWITCH=1` | Forces the offline state. |
| `SYRUS_DIAGNOSTICS=1` | Development only: adds sanitised token and provider diagnostics to turn responses (see `TOKEN_BUDGET.md`). |

The model is fixed by owner decision D2 (Groq `qwen/qwen3.8-27b`) in `configs/syrus_service.yaml`; there is no fallback, router or environment override. The production build has no mock mode: without a key the service is offline, and the evidence endpoints keep working. Tests inject a scripted double (`tests/test_syrus_service.py`); no test makes a network call.

## Endpoints

- `GET /api/health`: `ok` or `offline` with the reason; never the key.
- `GET /api/cases`, `GET /api/cases/{case_id}?mode=asof|retro`: the integrity-checked case files, read-only.
- `POST /api/session {case_id, mode}`: a signed 30-minute token bound to that case and mode, held with a server-side thread.
- `POST /api/syrus/turn {message, depth?, case_id?, mode?}` with `Authorization: Bearer <token>`: one retrospective turn. As-of sessions get `403 asof_chat_disabled`; a `case_id` or `mode` that disagrees with the token gets `409`.

## A turn

1. The server rejects as-of sessions, empty or over-500-character messages, a 16th turn, and a turn already in progress; it reports offline (no call) for a missing key, unconfirmed caps, the kill switch or an exhausted budget.
2. Context: the slice of the case's retrospective ledger the question needs (topics, required findings and their dependencies, always-present standing records; `selection.py`, `TOKEN_BUDGET.md`), the last two exchanges while they fit, the findings already stated, and any unverified user statements. The user's text is delimited as untrusted data. The structured-output schema carries only the offered IDs as enums, while validation uses the whole ledger. A prompt that is incomplete or over the token budget is refused (`context_incomplete`, `context_over_budget`) without a call.
3. One Groq call (`reasoning_effort: none`, strict JSON schema, 900 output tokens). The worst-case cost is reserved against the daily and total caps first and settled from the reported usage; a persisted spend file keeps the total across restarts.
4. The reply is parsed and checked (`validator.py`): schema and IDs; claim kind against cited record kinds; every number traced to a cited display string; signals, candidates and times only where a cited record carries them; absent sources only with the matching `unavailable:*` record; refused phrasing (cause, diagnosis, probability, certainty, prediction, healthy or false alarm, energy or cost, procedures, ranking, internal IDs) unless negated; formulas only at technical depth; the depth's unit budget; refusal categories that cite the record explaining the limit. Background units may not cite, count, name signals or refer to the case.
5. On a failed check the model gets one retry with the failed checks named, if at least 4 s remain of the 20 s turn. If a required finding is still missing the server appends its reviewed text under "Also relevant"; if the retry breaks a real check, the first reply (which passed everything but coverage) is used with the finding added. Otherwise the response is `failed_checks` with the check names and none of the model's text.
6. Assembly (`answer.py`): each unit becomes a block with its qualifications inline; the follow-up question comes from the test table; the operator note and provenance (case hash, model, cited-record count, validator result) are server-written.
7. The thread remembers stated findings, shown qualifications, asked tests and unverified statements. Logs hold turn metadata only: no message or answer text.

## Choices made while implementing the contract

- The server owns depth. The user picks `quick`, `investigation` or `technical`, or `auto`, which takes the deepest default among the declared purposes. The model does not report a depth.
- Required findings per purpose are limited to `what_happened` (margin, smoothing memory, and evidence timing from investigation depth) and `possible_causes` (candidates, identifiability, and repair non-uniqueness from investigation depth). Other purposes require none, so a mis-declared purpose can add at most a short "Also relevant" line.
- Context qualifications (repair scope, persistence rule, units, calibration week) appear once per conversation. Critical ones appear whenever their claim does, in short form after the first time.
- Triggers use the record and finding IDs a unit cites directly, plus patterns in the unit text in English and Vietnamese, plus the declared refusal category. Over-triggering shows a qualification the claim did not need; under-triggering is covered by the refused-phrasing and uncited-entity checks.
- Finding texts are English only, so "Also relevant" lines are English in Vietnamese answers (labelled as such). Qualification and server text are written in both languages; the Vietnamese needs owner review.
- A follow-up question is offered only when `possible_causes` was declared, no refusal was made, the test has not run or been asked, and the previous turn did not ask one.

## Not yet verified

- No live Groq call has been made. The model ID, strict-schema support and `reasoning_effort: none` for `qwen/qwen3.8-27b` must be checked at the first approved smoke test, together with the per-million prices in the config.
- Size against the provider limit: the first live call was refused (estimated input 8,162 plus 900 output against 8,000 tokens per minute). Prompts are now selected per question and capped at 5,100 estimated input tokens (`TOKEN_BUDGET.md`); the estimate itself is checked at the next approved live call. A second user inside the same minute can still be refused (`rate_limited`); a higher tier is a decision for the owner before public use.
- The acceptance questions (`tests/fixtures/syrus_acceptance.yaml`) have not been run against the model.
- Sessions, rate limits and the budget lock are in memory for one process; the cumulative spend is the only state persisted. Behind a proxy the client address needs `--proxy-headers` and trusted forwarding.
- The frontend is not built (Node is not installed); the app serves the API only.
