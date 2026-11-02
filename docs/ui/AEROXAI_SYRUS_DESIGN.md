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

**Investigation** is two coordinated panes. Evidence (left) is a short flow of collapsible sections with a sticky section nav: **Alert** (open: mechanics and the three most useful Verified Findings, with "View all N findings" opening every finding in full), **Why the detector alerted** (open: contributions with calibration ranks, and the evidence window as a chart of real percentiles), **What was tested** (counterfactual repairs), **What it could be** (unranked candidates, the leakage/demand limit, discriminating observations), **What is not known** (missing evidence, the physical-action gate, and the documented label in Retrospective mode only). Each collapsed header carries one figure. Syrus (right): depth chips with a one-line hint (Quick Note, Investigation, Technical, sent as `quick`, `investigation`, `technical`), verified highlights, suggested questions, and answers in which the server's qualifications appear inline as labelled "Evidence note" blocks, with numbered source markers on each claim and one collapsed "Sources" list per answer. Every marker, metric and finding opens the inspector with the exact text, caveat, source files and hashes, from the loaded case file only. Dialogs hold focus, close on Escape and return focus; only the top dialog answers Escape.

**Past-only mode** shows the evidence available at the alert and no retrospective label. Chat is off there (the service does not answer as-of questions); the pane shows verified highlights and says why. **Retrospective mode** enables chat.

**Phone and tablet** (up to 820 px): the page opens on Syrus with a one-line alert summary, incident and mode behind one "Change" control, and the chat pane sized to the viewport so the composer stays visible; Evidence is one tap away; the inspector is a bottom sheet.

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

## Refinement pass

Checked at 1440, 1920, 820 and 375 px. The default Investigation page went from about 4,700 px to about 2,100 px with the two open sections (about 1,500 px with all collapsed). Fixed-copy type is 13 px or larger except uppercase labels at 12 px; generated answers are 16 px. The main container uses `overflow: clip`, since `hidden` broke sticky positioning and let `scrollIntoView` scroll it.
