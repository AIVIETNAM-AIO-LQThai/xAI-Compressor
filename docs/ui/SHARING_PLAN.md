# Sharing plan

Decisions of 2026-11-03: private preview over a Cloudflare Quick Tunnel with the e-mail PIN, guest Syrus off by default; public showcase as a static export. Nothing has been deployed, tunnelled or uploaded. Every command below is for the owner to run, after reading the gates.

## Modes

| | `local` | `preview` | `public` (server) | showcase (static) |
|---|---|---|---|---|
| Set by | default | `AEROXAI_MODE=preview` | `AEROXAI_MODE=public` | `npm run build:showcase` + export |
| Who gets in | this machine | invite code (and the tunnel's e-mail PIN) | anyone | anyone |
| Syrus | live with your `.env` | off; live only with `AEROXAI_SYRUS_GUESTS=live` | no session or turn route, no key | no server, no key |
| Scenario Lab | live | live, 30 runs per guest per 10 minutes | not served | six labelled replays |
| Simulation Lab, Advisory | live | not served (nav hides them) | not served | not served |
| Data | full | full | full | precomputed, 176 MB |

How the server enforces it (`backend/access.py`, tested in `tests/test_access.py`):

- An unknown mode, `SYRUS_DIAGNOSTICS=1` outside local, or `preview` without `AEROXAI_PREVIEW_CODE_SHA256` stops the server at start.
- In `preview` every path except `/health` and the invite form needs a signed HttpOnly, SameSite=Strict cookie (12 hours); without it, pages show only the invite form and API paths answer `401 preview_locked`. Ten wrong codes per address per ten minutes, then `429`.
- Outside local, unless guests are allowed, Syrus is created with no key and no adapter object, so no code path can reach the provider; the interface shows "Chat is off here".
- Guests allowed: the daily cap becomes `AEROXAI_PREVIEW_DAILY_USD` (default $2, never above the owner's $10), each guest gets `AEROXAI_GUEST_TURNS` turns per UTC day (default 10), and the launcher refuses a second process on the same budget file.
- Per guest: 240 requests a minute, 30 lab runs per 10 minutes. Explorer windows stay bounded (14 days of 5-minute data, 24 hours of 10-second data).
- `scripts/run_preview.py` binds 127.0.0.1 only, one worker, forwarded addresses trusted from 127.0.0.1 only (a visitor cannot spoof their address through the tunnel; tested). No router port is opened.

## Private preview for judges (Quick Tunnel + e-mail PIN)

What it is: a random `https://….trycloudflare.com` address that exists while two processes run on this PC. Cloudflare describes Quick Tunnels as for testing and development, with no uptime guarantee, up to 200 in-flight requests and no server-sent events (AeroXAI uses none). Visitors need no Cloudflare account. The PC must stay on and awake.

### Gates before anyone gets the link

1. `npm run build` in `frontend` (the preview serves `frontend/dist`) and the explorer index exists (`data/processed/explorer/manifest.json`).
2. `cloudflared` is installed by you, from Cloudflare's official package, for example `winget install --id Cloudflare.cloudflared`.
3. You open the URL yourself on a phone using mobile data (not the home Wi-Fi): it asks for an e-mail PIN, then the invite code; after that `/config` shows `"mode":"preview"` and the Syrus drawer says chat is off.
4. Only then send the link, separately from the invite code.

### Commands (PowerShell, from the repository root)

Make an invite code and its hash (the code is shown once; only the hash goes into the environment):

```powershell
$code = .venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(12))"
$env:AEROXAI_PREVIEW_CODE_SHA256 = .venv\Scripts\python.exe -c "import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())" $code
Write-Host "Invite code to send to judges: $code"
```

Terminal 1, the server:

```powershell
$env:AEROXAI_MODE = "preview"
$env:PYTHONPATH = "."
.venv\Scripts\python.exe -m scripts.run_preview --port 8010
```

Terminal 2, the tunnel, with the judges' addresses (or `'*@their-university.edu'`):

```powershell
cloudflared tunnel --url http://127.0.0.1:8010 --allowed-mail judge.one@example.com --allowed-mail judge.two@example.com
```

`cloudflared` prints the `https://….trycloudflare.com` address. Run gate 3 before sharing it.

### Optional: live Syrus for one session

Only if you decide to; your Groq key pays, under the $2/day preview cap and 10 turns per guest per day. Stop the server, then in Terminal 1:

```powershell
$env:AEROXAI_SYRUS_GUESTS = "live"
.venv\Scripts\python.exe -m scripts.run_preview --port 8010
```

The key and `SYRUS_CAPS_CONFIRMED=1` still come from your server environment or `.env`; they never reach the browser. Turn it off again with `Remove-Item Env:AEROXAI_SYRUS_GUESTS` and a restart.

### QR code

Only for a URL that passed gate 3: open it in Chrome on this PC, choose "Share" then "Create QR code" in the address bar. Never encode `localhost`, `127.0.0.1` or a guessed address.

### Judge test flow (about five minutes)

1. Open the link, enter the e-mail PIN, then the invite code.
2. Overview: read the detector activity chart; open "Guided example: Incident 2" or any incident from the list.
3. Investigation: open a finding in the inspector; switch Past-only and Retrospective.
4. Data Explorer: pick an unlinked alert episode, click a bin, switch the channels to raw 10 s.
5. Investigate that episode: see the computed, unreviewed contributions.
6. Scenario Lab: run the tp3 offset (size 12) and the generic twin with "Run the same size as a demand step".

### Shut down, revoke, roll back

- Stop sharing now: Ctrl+C in Terminal 2. The address stops working at once and is never reused.
- Stop the app: Ctrl+C in Terminal 1. Guest cookies die with it (the signing secret is random per start unless `AEROXAI_PREVIEW_SECRET` is set).
- Revoke one judge: restart the tunnel without their address, and the server with a new code; the old link and cookies stop working.
- Syrus off at once without stopping anything else: restart with `$env:SYRUS_KILL_SWITCH = "1"`.
- A key may have leaked: revoke it in the Groq console and create a new one; nothing in the app needs changing.
- A stale lock after a crash (`outputs\syrus_budget.lock`): delete it only after checking no server is running.

## Public showcase (static export)

```powershell
cd frontend; npm run build:showcase; cd ..
.venv\Scripts\python.exe -m scripts.export_showcase
.venv\Scripts\python.exe -m http.server 8003 --bind 127.0.0.1 --directory outputs\showcase
```

The export fetches every read-only answer through the real application with Syrus keyless, writes one file per recorded day, six lab replays and `_headers` (no-sniff, no-referrer, deny framing, a same-origin content security policy), then refuses to finish if any file contains a credential name, a `gsk_` token or any `SYRUS_`, `AEROXAI_` or `GROQ` value from the environment. Result on 2026-11-03: 807 files, 176 MB, largest 1.4 MB, written in 26 s.

Checked locally: the joined day files equal the server's own answers for three windows crossing days and gaps (458, 368 and 367 rows) and a raw window (1,816 rows); the Simulation Lab and Advisory say they are not part of the showcase; the Syrus drawer says chat is off; the Evidence Proof page states that its integrity check was run at export time.

Upload, only after your approval and with your own Cloudflare account (free plan; within its 20,000 files and 25 MiB per file):

```powershell
npx wrangler pages deploy outputs\showcase --project-name aeroxai-showcase
```

The showcase carries the MetroPT-3 credit (CC BY 4.0 requires it): Davari, Veloso, Ribeiro & Gama (2021), MetroPT-3 Dataset, UCI Machine Learning Repository, doi:10.24432/C5VW3R. It redistributes derived data (5-minute summaries and 10-second samples); the licence allows that with credit. The recording can be reassembled from the day files; that is acceptable under the licence, so no bulk-export restriction is added.

## Paths compared

| | No-login viewing | Syrus, who pays | Scenario Lab | Data hosting | Cost and limits | Complexity |
|---|---|---|---|---|---|---|
| Quick Tunnel + e-mail PIN (chosen for the preview) | no | off by default; your key if enabled | live | this PC | free; testing-grade, no uptime guarantee, new URL each run | low |
| Named tunnel + Cloudflare Access | no | as above | live | this PC | free tier; needs a domain on Cloudflare | medium |
| Static export on Cloudflare Pages (chosen for public) | yes | none | replays | 176 MB of files | free | medium once, low after |
| Managed container (e.g. Render) | yes | possible only after a durable budget | live or replay | about 46 MB in the image | free tier sleeps after 15 min; 2 GB about $25/month (verify) | medium |

## Not ready

- Live Syrus for unknown visitors: the budget, sessions and governor are per process; a durable shared budget and abuse controls beyond the invite would be needed.
- Any multi-worker or multi-instance server with live Syrus.
- A stable judge URL: needs a named tunnel and a domain on Cloudflare.
- A real-phone check of the preview and the showcase.

## Sources

[Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/), [Cloudflare Access for public apps](https://developers.cloudflare.com/cloudflare-one/applications/configure-apps/self-hosted-public-app/), [Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/), [Render free instances](https://render.com/docs/free), [UCI MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset).
