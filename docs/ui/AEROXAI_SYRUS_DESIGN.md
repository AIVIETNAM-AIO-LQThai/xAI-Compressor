# AeroXAI + Syrus: interface design and integration

## From the pitch deck to the product

The deck supplies the brand: deep black, one vivid orange, white type, the Detect → Explain → Guide story. The product keeps that and drops the cinematic parts. Layout is graphite panels with an orange accent for the brand, the selected state and the primary action; green, amber and red appear only for real status. No machinery photography, no glow, no web fonts (system type, monospace only for values and IDs).

| Deck / concept | In the product |
|---|---|
| Slide 6 dashboard (health %, uptime, "78% confidence", inlet-valve cause) | Not reproduced. The investigation page shows only values from the case file. |
| Slide 7 phone copilot (valve position, manuals, PLC data) | Renamed Syrus. Cited answers from the loaded records; manuals, SOPs, raw telemetry and signal definitions are shown as "not loaded". |
| Slide 8 benchmark | Overview shows 4 / 4, 2 / 4 and 0.886 benchmark-unmatched episodes per 24 h, with the limits in the labels. |
| Concept mockups | Layout reference only. Every number on screen is bound to the API. |

## Navigation

Overview · Investigation · Detector Replay · Evidence Proof · Simulation Lab · Advisory. Only Investigation is new; the other pages keep their function and take the new theme. Routes are hash routes: `#investigation/<case>/<asof|retro>`.

**Investigation** is two coordinated panes. Evidence (left): score, threshold and margin; alert mechanics (the 3-of-4 bins, previous smoothed state against the raw score); Verified Findings; detector contributions with calibration ranks; the evidence window as a chart of real percentiles; counterfactual repairs; unranked candidates, the leakage/demand limit and the discriminating observations; missing evidence; the physical-action gate. Syrus (right): depth chips (Quick Note, Investigation, Technical, sent as `quick`, `investigation`, `technical`), verified highlights, suggested questions, claim blocks with the server's qualifications inline, "Also relevant" lines, a "Next useful question" from the test table, and chips that open the evidence inspector. Every chip, metric and finding opens the inspector with the exact text, caveat, source files and hashes for that record, from the loaded case file only.

**Past-only mode** shows the evidence available at the alert and no retrospective label. Chat is off there (the service does not answer as-of questions); the pane shows verified highlights and says why. **Retrospective mode** enables chat.

**Phone** (under 820 px): the page opens on Syrus with a one-line alert summary and compact incident and mode controls; Evidence is one tap away; the inspector is a bottom sheet.

## Integration

One origin: `uvicorn backend.web:app`. It serves `/api/*` (the Syrus service: one budget, one session store, one governor), the existing evidence, twin and optimisation routes, and `frontend/dist`. The browser and a phone use one URL, so there is no CORS and no `localhost` in a QR code. In development Vite serves the app and proxies the same paths to port 8000. Chosen over a reverse proxy because it needs no extra component, and over two uvicorn processes because two Syrus apps would each hold their own budget and sessions.

Sessions are created on the first question for a case and mode, held in memory only (nothing in browser storage), and kept per case and mode, so changing either never shows another conversation. A conversation survives moving to another page and back, and ends on reload or after the server's 30 minutes. The API key and caps flag never reach the browser; the bundle contains no `VITE_*` secret.

The governor admits one provider call at a time inside the provider's per-minute token window; a visitor who cannot be admitted sees "Syrus is answering someone else" with a retry countdown. This makes a shared demo safe; it is not a claim of concurrent capacity.

With `AEROXAI_PUBLIC=1` the simulation routes are not served at all.

## Status

| Feature | Status |
|---|---|
| Overview, Investigation, Syrus chat, inspector, past-only view | Working; verified against the backend with a scripted model double |
| Syrus answers from the real model in the interface | Not yet observed; the two earlier live calls were through the smoke script |
| Evidence values, findings, contributions, repairs, candidates | Real historical (MetroPT-3, 2020), read from the case files |
| Detector Replay, Evidence Proof | Working, real historical evidence |
| Simulation Lab, Advisory | Working, simulated; labelled as such |
| Manuals, SOPs, maintenance records, raw telemetry, signal definitions | Unavailable; shown as not loaded |
| Real-machine recommendations | Withheld |
| Multi-user QR demo | Not cleared: needs a phone test on the real network and a load test |
| Charts of raw score trajectories | Future; the case files carry no raw series |

## Trade-offs

- The finding titles and error messages are fixed UI text keyed by finding ID and error code; the finding and record text itself is shown as the ledger wrote it, in English.
- Quick Note is a depth setting for a real Syrus call, not a canned summary. The two verified highlights shown before the first question are ledger text, not generated.
- The Syrus pane keeps full answers and every qualification; only the fixed interface text is short.
