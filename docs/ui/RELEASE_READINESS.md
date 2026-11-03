# Release readiness

State on 2026-11-03, after the finalization stage (`FINALIZATION_PLAN.md`). "Done" means checked here, offline; "Pending" means it needs you, a real device or an authorised live call.

## Automated checks

| Check | Result |
|---|---|
| Backend tests, whole `tests/` folder | 773 passed. 21 modules not collected: 12 need `torch`, 9 need `pynvml`; neither is installed (research code, unchanged). They are not counted as passing. |
| New backend tests | `test_lab_twin.py` (10: separate streams, onset and duration, mass balance, nothing before onset, equal totals give equal responses, the controller adapter equals its own leak input, seeds), `test_access.py` (18: modes, invite gate, cookie tampering and expiry, guessing limit, guests off never reach the provider, guest quota and cap, launcher binding and lock, forwarded-address spoofing) |
| Frontend tests | 33 passed (new: twin comparison, static file mapping, day joining at a gap on a day boundary) |
| Ruff (`backend scripts tests ml`), TypeScript, production build, showcase build | clean |
| Static showcase against the server | identical answers for three cross-day windows and a raw window |
| Secrets | none in `frontend/dist`, `frontend/dist-showcase` or `outputs/showcase` (credential names, `gsk_`, environment values) |
| Frozen research artifacts | unchanged; `ml/` unchanged (the twin adapter is in `backend/app/lab.py`) |

One test was corrected, not loosened: `test_the_build_is_deterministic` compared Parquet bytes, which embed the writer's version; the index was written with pyarrow 26.0.0 / pandas 3.0.6 and this environment now has 25.0.1 / 3.0.5. It now compares the data exactly (and `episodes.json` byte for byte).

## Browser checks (emulated widths 1920, 1440, 820, 375)

Done on the scripted server and on the preview and showcase builds:

- every page: no horizontal overflow, one `h1`, page title, no text under 12 px, controls at least 40 px;
- nothing is selected by default; "Last chosen: Incident N" on pages without their own selection; Incident 2 appears only as a labelled guided example;
- generic twin: known inputs (legitimate demand, leakage, total) apart from the observable response; the same-size comparison computed from both runs; the form folds to one line after a run on a phone;
- preview: invite page only, HttpOnly cookie, Simulation Lab and Advisory hidden and explained, chat off, lab live behind the gate;
- showcase: replays labelled, proof check dated, chat off, no simulation or optimisation page.

Pending:

- a real phone (iOS Safari and Android Chrome) on the preview and the showcase; emulation is not device validation;
- keyboard-only pass on the new comparison table and replay list (structure is native buttons and table headers, not yet walked through by hand).

## Syrus

| Item | State |
|---|---|
| Validator strict, no bypass | Done; unchanged |
| Vietnamese acceptance question ("Có nhật ký bảo trì hoặc tài liệu kỹ thuật nào xác nhận…") | Offline fixture passes: a merged unit still fails `kind_mismatch`, a correctly split answer passes, the retry carries the repair hint. Pending: one authorised live call to see the model produce a passing answer. |
| `busy`, `paced`, rate-limited, offline, retry, no duplicate message | Offline tests pass; new states `guests_off`, `guest_quota`, `preview_locked` have messages and no pointless retry |
| Live answer in the new drawer | Pending: never observed |
| Off for episodes, intervals, simulations, preview guests (default) and the showcase | Done |

## Simulation honesty

- Generic twin labelled `SIMULATED · generic plant`, detector not applicable, energy simulated.
- Sensor-fault lab: a declared edit to recorded samples; for tp2, tp3 and reservoirs the form says the alert may reflect the broken relation between channels more than a fault; training-range counts with and without the change.
- The tp3 size-12 example alerts 20 minutes after onset: a detector sensitivity example, not fault prediction.

## Known limitations carried into this release

- Episode evidence at the first alert bin only (8 of 142 episodes change their leading group; deferred by decision).
- Contribution list shows all 15 groups.
- Units not loaded; document gate pending.
- Budget, sessions and governor are per process (see `SHARING_PLAN.md`, Not ready).

## Before sharing

1. Real-phone check of the preview (gate 3 in `SHARING_PLAN.md`).
2. Decide whether any session gets live Syrus; if yes, one authorised live call first, including the Vietnamese question.
3. For the showcase: review `outputs/showcase` locally, then approve an upload.
