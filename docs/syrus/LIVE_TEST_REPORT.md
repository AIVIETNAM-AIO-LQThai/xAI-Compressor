# Syrus live and offline test report

As of 2026-11-02. Each result is labelled by where it comes from: **observed** (output the owner supplied from their own live runs), **reported** (stated by an earlier working session, not re-run for this report), or **re-run** (executed in this working session, offline).

## Observed live calls (Groq `qwen/qwen3.8-27b`, owner-run, `scripts/syrus_smoke.py`)

| | `why_alert` | `definitely_a_leak` |
|---|---|---|
| Question | Why did this alert fire? | Was this definitely an air leak? |
| Case | Incident 2, 2020-05-29 | same |
| Final status / validator | `ok` / `passed` | `ok` / `passed` |
| Provider calls (HTTP 200) | 1 | 2 |
| Input tokens (actual) | 1,981 | 1,616 + 1,999 |
| Output tokens (actual) | 304 | 319 + 315 |
| Provider time | 3,719 ms | 3,797 + 2,109 ms |
| Estimated input, first request | 3,305 | 2,853 |
| Missing essentials | none | none |
| `refusal` | none (fixture: none) | none (fixture: `diagnosis_request`) |

The recorded spend file shows $0.0348 cumulative provider-reported spend and no calls of unknown outcome. That includes earlier runs.

Scope: two questions on one case. They do not establish other languages, other cases, multi-user throughput, answer quality in general, or physical correctness.

What the owner's review of the two answers found (the reading belongs to the owner, not to this report):

- `why_alert`: the figures, the smoothing memory and the 3-of-4 rule were stated correctly. The phrase "99.99% of the score came from earlier bins" can read as physical persistence, and the persistence rule appeared both in the prose and as a server qualification.
- `definitely_a_leak`: the label was attributed to the provider's list, leakage and downstream demand were stated as indistinguishable, and unloaded documents were acknowledged. The refusal metadata was empty although the fixture expected `diagnosis_request`, and the leakage/demand limit appeared twice. The reason the first of the two responses was not accepted was not visible in the diagnostics.

## Open items from those runs and what changed

| Item | Status |
|---|---|
| `refusal` empty on a diagnosis-certainty question | Fixed in code: a question the server classifies as leak certainty now returns `refusal: diagnosis_request` with `refusal_by: server`. A refusal the model chose is kept (`refusal_by: model`). Not yet re-observed live. |
| Retry reason invisible | Each call in the developer diagnostics now has `verdict` (`accepted`/`rejected`), the check codes and, for a missing finding, the finding IDs. No model text, no check details. Not yet re-observed live. |
| Duplicate leakage/demand limit, duplicate 3-of-4 rule | A qualification is no longer appended beside a unit whose own text states the same thing. Not yet re-observed live. |
| "came from earlier bins" wording | The finding now reads "the smoothing state carried from earlier bins accounts for 99.99% of the current smoothed score". The case files were rebuilt (manifest `9aaadf859451`). |
| Concurrent visitors | A shared governor admits one provider call at a time within 90% of the tokens-per-minute window; others get reason `busy` and a retry time. Tested with scripted doubles only; not load-tested. |

The earlier `429` and `httpx.ReadError` failures were followed by successful calls. The cause of the `ReadError` is not established.

## Re-run offline in this session

| Check | Result |
|---|---|
| `pytest` on `tests/test_syrus_{integration,label,guards,context,service,cases}.py` | 394 passed |
| `ruff check backend scripts tests` | clean |
| `python -m scripts.syrus_tokens --all-cases` | 76 of 76 question-turns have every essential finding offered; largest estimated input 4,570 against the 5,100 cap; no dropped topics |
| `python -m scripts.syrus_smoke` (dry run) | exits 0; plan and worst-case cost only |
| Frozen research artifacts | `git diff` on `docs/*.json`, `docs/research`, `ml/`, `notebooks/` is empty |

These checks use scripted model doubles and mock transports. They show that the evidence is offered, the rules are enforced and the plumbing works. They do not show live answer quality.

## Not yet tested live

Counterfactual causal questions, unsupported savings questions, Vietnamese document requests, multi-turn follow-ups, the new refusal label, retry verdicts, the governor, and the single-origin app with a phone. Each paid call needs the owner's approval:

```powershell
python -m scripts.syrus_smoke --run --only dv_pressure_cause
python -m scripts.syrus_smoke --run --only energy_wasted
python -m scripts.syrus_smoke --run --only maintenance_log_vi
```

## Claims boundary

MetroPT-3 is a historical metro air-production-unit compressor, not a live plant. No real-asset control action, measured saving or confirmed root cause is established. The separate physics and optimisation results are synthetic.

## Interface checks (re-run, scripted model double)

Run in the built-in browser against `backend.web` with `scripts/qa_web.py` (a scripted stand-in for Groq; the interface showed its "Test fixture" banner). These show the interface and the plumbing, not Syrus's answers.

| Check | Result |
|---|---|
| Overview from the real evidence and case APIs: 4 / 4, 2 / 4, 0.886, four incident cards with pre-/post-onset tags | Rendered correctly |
| Investigation, Incident 2 and 3, retrospective: metrics, alert mechanics, findings, contributions, chart, repairs, candidates, missing evidence | Rendered from the case file; no overflow at 1440 px |
| Question sent, answer rendered with claims, inline qualification, evidence chips, validator chip, "No diagnosis claimed" | Worked |
| Chip opens the inspector with text, rule and inputs; Escape and the close button close it | Worked |
| Depth chips change the depth sent (unit test confirms `quick` / `investigation` / `technical`) | Worked |
| Past-only mode: no label text, no chat box, a reason shown; the retrospective conversation was still there on return | Worked |
| Failure states in the browser: rate limited with countdown, busy with countdown, failed checks with the check codes, offline | Worked; a disabled "Try again" during a countdown |
| Existing pages (Detector Replay, Evidence Proof, Simulation Lab, Advisory) through the proxy | Loaded without errors, new theme |
| Phone width (375 px): no horizontal overflow, chat first, summary strip, one tap to Evidence | Worked; inline text links are smaller than a touch target |
| Frontend `npm run build`, `tsc`, and `npm test` (10 client and message tests) | Passed |
| Secret scan of `frontend/dist` | No key patterns found |

Found and fixed during these checks: the page crashed on the first answer (an effect returned a value instead of a cleanup); a limit stated in one unit was still appended beside another unit; a provider refusal kept its tokens in the governor's window; the governor reserved the pessimistic estimate and refused a second question within a minute (it now learns the real ratio, never below half).

Not checked: a real phone on a network, two simultaneous visitors, Vietnamese in the interface, keyboard-only use beyond focus outlines and Escape, screen readers.
