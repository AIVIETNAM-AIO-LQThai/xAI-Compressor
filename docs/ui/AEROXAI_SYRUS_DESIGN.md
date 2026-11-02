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

Overview · Investigation · Data Explorer · Detector Replay · Evidence Proof · Simulation Lab · Advisory. Routes are hash routes. An incident is `#investigate/incident/<case>/<asof|retro>`; the earlier `#investigation/<case>/<mode>` links still open and are rewritten to the current form.

**Nothing is selected by default.** `#investigate` without an incident shows a picker: the four documented incidents and, when this tab has one, "Continue with" the last incident the user chose. The Overview offers the same incidents and a labelled "Guided example: Incident 2". The selection lives in the URL; the last explicit choice is also kept in `sessionStorage` (case and mode only, never a conversation), so the Investigation link and Syrus return to it.

**Overview** leads with detector activity over the whole recording, drawn from the explorer index: the highest smoothed score per 6 hours on a log axis, the frozen threshold, periods that contain an alert, training and calibration shaded (alerts are not evaluated there), gaps left blank, and the four documented incidents marked as retrospective. A monthly table carries the same numbers. Below it: recorded span, valid-bin share, alert episodes (linked and not linked to a documented incident), and the benchmark as evaluation context. Without the index, the page says how to build it and still lists the incidents.

**Data Explorer** (`#explore`) shows the recorded data for what the user chooses: an alert episode (`#explore/episode/<start>`), a documented incident window, or any interval of up to 14 days (`#explore/interval/<start>/<end>`, dataset local time). Nothing is shown until one is chosen. The page has the detector score (smoothed and raw, the frozen threshold, alert periods, training and calibration shaded, documented incidents marked as retrospective, smoothing restarts marked, gaps left blank); a click on the chart opens that bin with its score, state, sample count and the seven analog channels' mean, minimum and maximum, and offers a 6-hour zoom around it; and all 15 channels, as five-minute summaries or, for windows of 24 hours or less, as the recorded 10-second values. Values are as recorded and the charts say "units not loaded" until the dataset description passes the document gate. An episode linked to an incident offers "Investigate Incident N"; an episode not linked says it is not linked, is not a confirmed false alarm and has no Syrus case file. Syrus is off for episodes and intervals and never borrows an incident's findings.

**Investigation of an alert episode** (`#investigate/episode/<start>`, from the Data Explorer) is computed on request by `backend/app/episode_evidence.py` with the frozen detector's own explanation code, so it needs no case file. It shows the smoothed score against the threshold at the first alert bin, the detector contribution of each signal group with its calibration percentile, how the 3-of-4 rule was met (restarting after a gap), and the percentile of the three leading groups over the preceding hour. The page is labelled "Computed · not reviewed · Syrus off" and says what it does not compute: counterfactual repairs, candidate explanations, documents. The same code reproduces the ranking, shares, percentiles and evidence window of all four incidents in `docs/xai_verification_report.json` (tested). It makes no causal claim, and an unlinked episode is never called a false alarm. A linked episode offers the reviewed incident.

**Investigation** is the evidence for one selected incident, full width. Evidence is a short flow of collapsible sections with a sticky section nav: **Alert** (open: mechanics and the three most useful Verified Findings, with "View all N findings" opening every finding in full), **Why the detector alerted** (open: contributions with calibration ranks, and the evidence window as a chart of real percentiles), **What was tested** (counterfactual repairs), **What it could be** (unranked candidates, the leakage/demand limit, discriminating observations), **What is not known** (missing evidence, the physical-action gate, and the documented label in Retrospective mode only). Each collapsed header carries one figure. **Syrus** is a still launcher at the bottom right that names its context ("Incident 2 · Retrospective" or "No incident selected") and opens a drawer: beside the page from 1680 px, over it below that, full screen on a phone. Its context is the incident on screen, or elsewhere the last one the user chose; without either it lists the incidents to choose from and says that episodes, intervals and simulations have no reviewed Syrus contract yet. "Ask Syrus about this" in the inspector places a question about that finding or record in the composer and never sends it. Inside the drawer: depth chips with a one-line hint (Quick Note, Investigation, Technical, sent as `quick`, `investigation`, `technical`), verified highlights, suggested questions, and answers in which the server's qualifications appear inline as labelled "Evidence note" blocks, with numbered source markers on each claim and one collapsed "Sources" list per answer. Every marker, metric and finding opens the inspector with the exact text, caveat, source files and hashes, from the loaded case file only. Dialogs hold focus, close on Escape and return focus; only the top dialog answers Escape.

**Past-only mode** shows the evidence available at the alert and no retrospective label. Chat is off there (the service does not answer as-of questions); the pane shows verified highlights and says why. **Retrospective mode** enables chat.

**Phone and tablet** (up to 820 px): incident and mode sit behind one "Change" control, the Syrus drawer fills the screen with the composer always visible, and the inspector is a bottom sheet. The Overview keeps its chart first and its four figures two across.

## Integration

One origin: `uvicorn backend.web:app`. It serves `/api/*` (the Syrus service: one budget, one session store, one governor), the existing evidence, twin and optimisation routes, and `frontend/dist`. The browser and a phone use one URL, so there is no CORS and no `localhost` in a QR code. In development Vite serves the app and proxies the same paths to port 8000. Chosen over a reverse proxy because it needs no extra component, and over two uvicorn processes because two Syrus apps would each hold their own budget and sessions.

Sessions are created on the first question for a case and mode, held in memory only (no conversation is written to browser storage), and kept per case and mode, so changing either never shows another conversation. A conversation survives moving to another page and back, and ends on reload or after the server's 30 minutes. The API key and caps flag never reach the browser; the bundle contains no `VITE_*` secret.

The governor admits one provider call at a time inside the provider's per-minute token window; a visitor who cannot be admitted sees "Syrus is answering another question." while a call is in flight, or "This minute's model allowance is used." when the window is full, with a retry countdown. This makes a shared demo safe; it is not a claim of concurrent capacity.

With `AEROXAI_PUBLIC=1` the simulation routes are not served at all.

## Status

| Feature | Status |
|---|---|
| Overview activity chart, incident picker, Investigation, Syrus drawer, inspector, past-only view | Working; verified against the backend with a scripted model double |
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
