# Redesign QA

Status: stages 1 to 7 of the product redesign (see `REDESIGN_FEASIBILITY.md`), checked on 2026-11-03. Interface checks ran against `scripts.qa_web`, which uses a scripted model double; no live model call was made and `.env` was not read.

## Automated checks

| Check | Result |
|---|---|
| Backend tests (`tests/`, whole folder) | 746 passed; 21 test modules do not collect because they import `torch` or `pynvml`, which are not installed here (research modules, unchanged) |
| Ruff (`backend scripts tests ml`) | clean |
| TypeScript (`tsc -b`) | clean |
| Frontend tests (`node --test tests/`) | 27 passed |
| Production build | passes; the single JavaScript chunk is about 1.3 MB (ECharts), as before |
| Explorer, episode evidence and lab tests | reproduce the frozen threshold, the Incident 2 score, the 142 episodes and all four incidents' frozen contribution rankings; the lab's unchanged background reproduces the explorer index |
| Frozen research files | untouched: no change under `docs/*.json`, `docs/research`, `ml/`, `notebooks/` or the frozen configs |
| Public build (`AEROXAI_PUBLIC=1`) | serves no `/twin`, `/optimization` or `/lab` route (tested); the lab page says it is not served |
| Secrets | no key, `SYRUS_*` name or `VITE_*` secret in `frontend/dist`; the only `VITE_` variable is the API base URL |

## Scripted browser sweep

Eleven routes (overview, investigation picker, incident retrospective and past-only, episode investigation, explorer chooser and episode, scenario lab, simulation lab, detector replay, evidence proof, advisory) at 1920, 1440, 820 and 375 px:

- no page-level horizontal overflow at any width; no element outside the viewport at 375 px (two earlier pages, detector replay and evidence proof, had one chip each running 18 to 27 px past the edge and now wrap);
- exactly one `h1` per page and a page-specific document title;
- no text under 12 px in the main area (the earlier pages used 9 to 11 px labels; they are now 12 px);
- no interactive control under 40 px except inline text links;
- no health score, remaining useful life, failure probability or "predicted" wording. The words "probability", "false alarm" and "healthy" appear only in negations ("not a fault probability", "not a confirmed false alarm", "not known to be healthy"). One earlier label, "Predicted next pressure", now reads "Simulated pressure at the horizon end";
- charts carry an `aria-label` and the Overview chart has a monthly table with the same numbers.

## Before and after

Before is commit 402ffbc, built and served with the same scripted model double.

| | Before | After |
|---|---|---|
| Overview | benchmark figures and four step cards; a button "Ask Syrus about Incident 2" hard-wired to one incident | the whole recording as a chart first (detector score, threshold, alerts, split periods, gaps, documented incidents as retrospective), then the figures; incidents and an explicit "Guided example: Incident 2" |
| Investigation without a link | opened Incident 2 silently | asks which incident; offers "Continue with" the last choice; never defaults |
| Syrus | a column inside Investigation, tabs on a phone, an animated mark | a still launcher naming its context, a drawer (beside the page from 1680 px, over it below, full screen on a phone), off on simulations and for episodes without a case file |
| Data | four incident case files only | any alert episode or recorded interval, raw 10 s channels, bin detail, gaps kept as gaps |
| Episodes without a case file | not reachable | computed, labelled "not reviewed", Syrus off |
| Simulation | constant-flow receiver only | Scenario Lab: declared sensor faults on a recorded background scored by the frozen detector, and a generic receiver twin with the detector marked not applicable |

Screenshots for each stage were produced during the work and are not committed.

## Limitations and risks

- Episode evidence is computed at the first alert bin only. Over all 142 episodes the three leading groups match at the peak bin in every case and the leading group in 138 (97%); 8 episodes change their leading group during the episode and 5 of those have a first-bin leader for under half the bins. The page labels the bin but does not yet show the peak or a consistency line. Deferred by decision.
- The contribution list shows all 15 groups; most carry under 0.1%.
- Scenario Lab changes are declared edits to recorded samples, not physical fault models; a change to one pressure channel also breaks the relation to its neighbour, which the detector reads. "No alert" means the detector ignored that edit, not that a real fault would go unseen.
- Chart units are not loaded; sizes in the lab are in training standard deviations.
- Charts were not tested on a real touch device; tooltips are on hover and tap.
- The detector is refit from the frozen training split on first use (about 0.4 s); results match the frozen values to the tolerances tested, not bit for bit.
- The raw-sample index stores float32.
- No live Syrus answer was observed in the new drawer; all answers in the checks came from the scripted double. The earlier live-test report remains the only evidence of real model behaviour.
- A multi-user demo has not been load-tested.
- Document approval for units and for the reviewed wording is still pending.
