# Finalization plan: twin accounting, product acceptance, safe sharing

Status: review and plan, 2026-11-03. Decisions taken the same day: guest Syrus off by default in the preview (switchable per session under guest quotas and a $2/day preview cap); private preview over a Quick Tunnel with the e-mail PIN; public showcase as a static export, prepared but not uploaded. Stages 1 to 5 of section 5 are implemented and validated offline (see `RELEASE_READINESS.md` and `SHARING_PLAN.md`); the static showcase is exported locally. Nothing is deployed, tunnelled or uploaded. Facts below were checked read-only against commit 73a999a and the local data; external facts against the official pages listed at the end.

## 1. What the review confirmed or corrected

| Claim in the brief | Finding |
|---|---|
| `Lab.twin()` adds the same extra outflow to the demand series for both `demand_step` and `leak_onset`, with `leak_mass_flow_kg_s=0.0` | Confirmed (`backend/app/lab.py`). The returned `demand_kg_s` is really total outflow, and a leak run reports its leak as demand. |
| Equal total outflow gives identical dynamics in the current model | Confirmed in the code, not just in results: the controller's set point uses `demand + leak` (`ml/control/baseline.py`, desired flow) and the receiver uses `m_in − m_demand − m_leak` (`ml/twin/physics.py`). Both depend on the sum only. |
| `AEROXAI_PUBLIC=1` still mounts the Syrus `/api/*` routes | Confirmed (`backend/web.py`). Nuance: calls are still bounded by the app caps ($10/day, $30 total), the one-call governor and 20 sessions per IP per hour. That bounds the bill, not the abuse: one anonymous visitor can spend the whole daily cap and lock everyone else out, and then the total cap in three days. |
| `.env` holds the key | `settings.py` loads the repository `.env` at import. A preview served from this machine is therefore live-capable whenever the key and caps flag are set, unless the preview mode turns Syrus off explicitly. |
| Single-process caps may not hold across workers | Confirmed. The budget is a JSON file read once at start and written by whichever process settles last; sessions and the governor are in memory. Two workers would each think they own the whole cap. Restarts are safe for the cumulative cap (the file persists) but end all sessions. |
| Visitor identity behind a tunnel | `cloudflared` connects from 127.0.0.1. Uvicorn 0.52 trusts forwarded headers from 127.0.0.1 by default and takes the right-most untrusted address, so per-IP limits see the real visitor and a client cannot spoof it by sending its own `X-Forwarded-For`. Worth a regression test because a changed `--forwarded-allow-ips` would break it. |
| `frontend/dist` alone does not provide the Explorer | Confirmed. Serving needs about 46 MB of git-ignored data: `data/processed/explorer` (30 MB, 212 daily raw shards), `features.parquet` (7.7 MB), `train/calibration/test.parquet` (8 MB). The 209 MB raw CSV is needed only to rebuild the index (about 7 s). Tracked in Git: `docs/*.json` (1 MB), `docs/syrus/cases` (1 MB). |
| Runtime resources | One process uses about 175 MB idle and about 365 MB after loading the explorer, the episode evidence model and the lab model; each loads in under 0.4 s. |
| Earlier pages break in public mode | Confirmed: Simulation Lab and Advisory call `/twin` and `/optimization`, which the public build removes; the nav still links to them. |
| Vietnamese `kind_mismatch` | The exact acceptance sentence is already an offline fixture (`tests/test_syrus_documents.py`, `VI`). Offline tests show a merged label-plus-candidate unit still fails `kind_mismatch`, a correctly split answer now passes (the label unit names the candidate it echoes), and the retry message carries the repair hint. What is not shown: that the live model now produces a passing answer. That needs one authorised live call, so bilingual readiness stays "validator fixed, model behaviour unconfirmed". |
| Stage-7 status | Accurate as stated: 746 passed, 21 research modules not collected (torch/pynvml absent), 27 frontend tests passed, no live Syrus answer in the new drawer, browser emulation only. |
| MetroPT-3 licence | CC BY 4.0 (UCI). Redistribution and screenshots are allowed with credit: Davari, Veloso, Ribeiro & Gama (2021), MetroPT-3 Dataset, UCI Machine Learning Repository, doi:10.24432/C5VW3R. The app shows the DOI and licence but no full citation yet. |

## 2. A. Generic receiver twin: accounting fix (no decision needed)

- **Inputs kept separate**, each a 1 s series over the run: base legitimate demand (seeded noise), added legitimate demand, leakage, and total outflow = their sum. `demand_step` changes only added demand, `leak_onset` only leakage, `none` neither. New optional `duration_minutes` (absent: to the end).
- **Dynamics:** the frozen controller is called unchanged with total outflow as its demand profile and zero scalar leak. That is exact for this model because both the controller and the receiver use only the sum, and a test proves it: for a constant leak, `simulate_baseline(D, leak=L)` equals `simulate_baseline(D + L, leak=0)` to floating-point tolerance. No change under `ml/`.
- **Response:** explicit `units`, `time_seconds`, `base_demand_kg_s`, `added_demand_kg_s`, `leak_kg_s`, `total_outflow_kg_s`, `pressure_bar_g`, `compressor_flow_kg_s`, `power_kw`, per-step `safety_violation`, compressor on-states, and a summary (energy, starts, violations, pressure range). The ambiguous `demand_kg_s` is removed rather than kept with a changed meaning; the only client is the Scenario Lab, updated in the same change.
- **UI:** "Known simulated inputs" chart (legitimate demand, leakage, total outflow) and "Observable plant response" charts (pressure with band, compressor flow, power). The equivalent-scenario button runs the other kind with the same size and shows both side by side: the inputs differ, the observable trajectories coincide, computed from the two responses. Wording limits the claim to this model and to observations of pressure and total flow. Label: `SIMULATED · generic plant`; detector not applicable.
- **Tests:** zero added demand and leak for `none`; each scenario changes only its own stream; nothing differs from baseline before onset; mass balance (total = parts) and units; same seed reproducible, other seeds change only the noise; equal-total runs match in pressure, flow, power, energy, starts and safety state; the frontend comparison text comes from the two responses.

## 3. B. Product acceptance (routine, no decision needed)

- **Source labels** as one small component used everywhere: `RECORDED` (MetroPT-3 data), `MODEL INFERENCE` (detector scores, contributions), `SIMULATED` (lab, twin, advisory), `UNREVIEWED` (computed episode evidence). Reviewed incident findings keep their own "verified" label.
- **Syrus context honesty:** on pages without their own selection the launcher currently shows the last incident chosen in this tab ("Incident 2 · Retrospective" on the Overview). It will say "Last chosen: Incident 2" so it never looks like a default.
- **No dead pages per mode:** the server reports its mode and features; the nav hides or replaces what the mode does not serve. The older Simulation Lab duplicates the Scenario Lab's generic twin and will leave the nav (route kept for links).
- **Scenario Lab:** the form collapses to a one-line summary after a run on phones; the sensor-fault note on pressure channels says that shifting one pressure sensor also breaks its relation to its neighbour, which the detector reads.
- **Audit and fix** keyboard focus, close and reopen state, loading, error and empty states, long results, legends and tooltips at 375 px, on every page in one pass.
- **Kept deferred, as decided:** the episode evidence peak-bin view and consistency line; listed as a known limitation in the release checklist.
- **Syrus UX states** (`busy`, `paced`, rate-limited, offline, retry, no duplicate message) re-tested offline in the drawer.

## 4. C. Safe sharing

### Server changes needed before any sharing (no decision needed; the defaults depend on decisions D1 to D3)

1. **One mode switch**, `AEROXAI_MODE=local|preview|public` (`AEROXAI_PUBLIC=1` kept as an alias for `public`). Unknown values refuse to start.
2. **Syrus gated on the server, before any provider call or session:** `public` does not mount `/api/session` or `/api/syrus/turn` at all (reviewed case files stay readable); `preview` mounts them only when `AEROXAI_SYRUS_GUESTS=live` is set, otherwise they answer `syrus_disabled` without touching the provider. `local` is unchanged. Negative tests for each mode.
3. **Preview gate in the app, fail-closed:** `preview` refuses to start without an invite secret (`AEROXAI_PREVIEW_CODE_SHA256`). Visitors enter the code once; the server sets a signed, HttpOnly, SameSite=Strict cookie; every API route checks it. This is defence in depth under the Cloudflare email check, and it means a mistyped tunnel command cannot expose the app.
4. **Quotas for guests:** per-visitor turn quota and a lower preview daily cap (`AEROXAI_PREVIEW_DAILY_USD`, default $2), plus a simple per-visitor rate limit on explorer and lab requests.
5. **Single process enforced** when Syrus is live outside `local`: an exclusive lock next to the budget file; a second process refuses to start. Documented `--workers 1`.
6. **Refuse unsafe combinations:** `SYRUS_DIAGNOSTICS=1` outside `local`; binding to anything but 127.0.0.1 in `preview`.
7. **Credit line** for the dataset in the footer and the Overview source note.

### Modes

| | Local (now) | Private preview for judges | Public showcase |
|---|---|---|---|
| Who | you | invited e-mail addresses | anyone |
| Data | full | full, read from this machine | precomputed, read-only |
| Syrus | live with your `.env` | off by default; live only if you set it, under guest quotas | off |
| Scenario Lab | live | live (bounded inputs, gated) | replays of fixed, labelled runs |
| Earlier Simulation Lab and Advisory | live | live | replays or hidden |

### Paths compared

| | (a) No-login viewing | (b) Syrus, who pays | (c) Scenario Lab | (d) Data hosting | (e) Cost and limits | (f) Complexity |
|---|---|---|---|---|---|---|
| **Quick Tunnel with e-mail PIN** (`cloudflared tunnel --url … --allowed-mail …`) | no: e-mail one-time PIN, no Cloudflare account for the visitor | off by default; live = your Groq key, app caps | live | your machine; it must stay on | free; random `trycloudflare.com` URL each run; Cloudflare says testing and development only, no uptime guarantee, 200 in-flight requests, no SSE | low: one command, no account, no domain |
| **Named Cloudflare Tunnel + Access** | no: Access policy (e-mail allowlist) | as above | live | your machine | Zero Trust free tier (under 50 users per Cloudflare); needs a domain active on Cloudflare (a purchase if you have none); stable hostname | medium: account, domain, tunnel, Access app, service install |
| **Static export on Cloudflare Pages** | yes | none: no server, no key exists there | replays only | about 190 MB of precomputed JSON (≈570 files; gzip on the wire about 15–20 MB): every page, every day of 5-min and 10-s data, 142 episodes, replays | free; within the free plan's 20,000 files and 25 MiB per file; no cold start; nothing to attack but files | medium once (an export script and a static data layer in the frontend), then low |
| *Managed container (e.g. Render)* | yes, or behind a gate | possible, but needs the durable budget work first | live or replay | data baked into the image (≈46 MB) | free tier sleeps after 15 min idle, about 1 min cold start, ephemeral disk; paid instances from 512 MB (tight at 365 MB working set); 2 GB tier about $25/month (third-party figure, check the live page) | medium |

Hugging Face Docker Spaces now need a paid PRO plan ($9/month) to create, so they are no longer a free option.

### Recommendation

- **Private preview: Quick Tunnel with the e-mail PIN, on top of the app's preview gate, Syrus off by default.** It needs no domain, account or open router port, it ends when you stop it, and the e-mail list plus the invite code make an accidental open link unlikely. Its limits suit a judging session, not a standing service. If judges need the same URL for days, move to a named tunnel with Access, which needs a domain.
- **Public showcase: a static export on Cloudflare Pages.** The recording is historical and frozen, so every public view is a pure function of fixed files; a static site serves it with no server, no key and no compute endpoint to protect, and it cannot masquerade as live simulation because there is none: lab entries are labelled replays of fixed runs. If you later want live Syrus for the public, that needs the durable budget and a container, and a separate decision.

### Not ready, whatever the decisions

- live Syrus for unknown visitors (needs a durable, cross-process budget and abuse controls beyond per-IP);
- any multi-worker or multi-instance deployment with live Syrus;
- a real-phone test (only browser emulation so far);
- a live Syrus answer in the new drawer, including the Vietnamese acceptance question;
- reviewed case files for unlinked episodes.

## 5. Order of work

1. Twin accounting and its tests; source labels; launcher wording; Scenario Lab layout.
2. Mode switch, server gates, preview gate, quotas, single-process lock, negative tests; mode-aware nav.
3. UX audit pass and fixes; offline Syrus state checks; dataset credit.
4. Offline validation (all suites, build, browser sweep at 1920/1440/820/375, built-file and network inspection for secrets).
5. `RELEASE_READINESS.md` and `SHARING_PLAN.md` with PowerShell commands for the preview (to run only after you approve), shutdown and rollback, judge test flow, QR steps for a verified HTTPS URL.
6. If the static showcase is chosen: export script, static data layer, replays, local preview of the static build. No upload until you approve.

## Sources

- [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
- [Cloudflare Access for self-hosted public apps](https://developers.cloudflare.com/cloudflare-one/applications/configure-apps/self-hosted-public-app/)
- [Cloudflare Access plans](https://www.cloudflare.com/en-au/products/cloudflare-access)
- [Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/)
- [Render free instances](https://render.com/docs/free), [Render instance types](https://render.com/docs/compute-plans)
- [Hugging Face Spaces overview](https://huggingface.co/docs/hub/spaces-overview), [Hugging Face pricing](https://huggingface.co/pricing)
- [UCI MetroPT-3 dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
