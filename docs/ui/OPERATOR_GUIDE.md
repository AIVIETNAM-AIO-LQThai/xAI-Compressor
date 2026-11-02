# Running AeroXAI + Syrus

## One-time setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
cd frontend
npm ci
```

Install Node.js LTS first (`winget install OpenJS.NodeJS.LTS`). If you moved the project from another operating system, run `npm ci` again so the platform binaries are right.

## Environment

Set in `.env` at the repository root (never committed, never read by the browser). Names only:

| Variable | Purpose |
|---|---|
| `SYRUS_LLM_API_KEY` | Groq key. Without it Syrus is offline and the rest of the app works. |
| `SYRUS_CAPS_CONFIRMED=1` | Confirms the spending caps in `configs/syrus_service.yaml`. |
| `SYRUS_SESSION_SECRET`, `SYRUS_IP_SALT` | Optional; fixed values keep sessions across restarts. |
| `SYRUS_KILL_SWITCH=1` | Forces Syrus offline. |
| `SYRUS_DIAGNOSTICS=1` | Development: token and provider diagnostics in responses. |
| `AEROXAI_PUBLIC=1` | Do not serve the simulation routes. |

`.env` is loaded automatically, so starting the app with a real key configured makes every question a paid call. To look at the interface without spending, start it with the kill switch (the interface shows Syrus as switched off) or use the interface check server below.

## Recorded-data index (once, and after any change to the frozen data)

```powershell
python -m scripts.build_explorer_index
```

Reads the local MetroPT-3 recording and frozen splits, rebuilds the frozen detector's series and writes `data/processed/explorer/` (about 30 MB, gitignored). It stops without a manifest unless it reproduces the frozen threshold, the Incident 2 smoothed score and the 142 / 133 episode counts. Without it the explorer endpoints answer `explorer_index_missing`.

## Start

Production-style, one origin:

```powershell
cd frontend; npm run build; cd ..
python -m uvicorn backend.web:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

Development, with hot reload (two terminals):

```powershell
python -m uvicorn backend.web:app --port 8000
cd frontend; npm run dev
```

Open `http://127.0.0.1:5173`; Vite forwards `/api`, `/evidence`, `/twin`, `/optimization` and `/health` to port 8000.

POSIX: the same commands with `.venv/bin/python` and `npm`.

## On a phone

Both devices on the same trusted network. Find this computer's address (`ipconfig`, the IPv4 line), start with `--host 0.0.0.0`, allow port 8000 in the Windows firewall prompt, and open `http://<that address>:8000` on the phone. That exposes the paid Syrus endpoint to everyone on that network; the caps, the per-address session limit and the governor still apply. A QR code must encode that same URL, never `127.0.0.1`. For an internet-facing demo use an HTTPS host and `AEROXAI_PUBLIC=1`; nothing here deploys anything.

## Checking the interface without a model

```powershell
python -m scripts.qa_web --port 8000
```

Starts the real app with a scripted stand-in for Groq. The interface shows a "Test fixture" banner. Typing `qa rate`, `qa offline`, `qa failed` or `qa slow` triggers the matching state. Its answers prove nothing about Syrus.

## Checks

```powershell
python -m pytest tests/test_syrus_integration.py tests/test_syrus_label.py tests/test_syrus_guards.py tests/test_syrus_context.py tests/test_syrus_service.py tests/test_syrus_cases.py -q -p no:cacheprovider
python -m ruff check backend scripts tests
python -m scripts.syrus_tokens --all-cases
cd frontend; npm test; npm run build
```

Paid check, one call, run by you: `python -m scripts.syrus_smoke --run --only why_alert`. Or ask the same question in the browser.

## Not yet established

A real Syrus answer rendered in the browser; the phone on the real network; two visitors at once; Vietnamese answers in the interface; the later smoke questions listed in `docs/syrus/LIVE_TEST_REPORT.md`.
