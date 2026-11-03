"""Start the invite-only preview the safe way: preview mode, 127.0.0.1 only, one worker, proxy headers trusted from 127.0.0.1 only.

    $env:AEROXAI_MODE = "preview"
    $env:AEROXAI_PREVIEW_CODE_SHA256 = "<hex SHA-256 of the invite code>"
    python -m scripts.run_preview [--port 8010]

The server listens on 127.0.0.1 only; a tunnel on the same machine is the only way in. When guests may use live Syrus, a lock file next to
the budget file makes a second server refuse to start, so two processes never spend against the same caps without seeing each other.
"""
import argparse
import atexit
import os
import sys
from pathlib import Path

import uvicorn

from backend.access import resolve
from backend.syrus_app.settings import load_settings


def main() -> None:
    ap = argparse.ArgumentParser(description="Invite-only AeroXAI preview on 127.0.0.1.")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()
    access = resolve()
    if access.mode != "preview":
        sys.exit("Set AEROXAI_MODE=preview first; this launcher only starts the preview.")
    lock: Path | None = None
    if access.syrus_guests:
        lock = load_settings().budget_file.with_suffix(".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            sys.exit(f"Another server holds {lock}. Stop it first; if none is running, delete the file.")
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        atexit.register(lambda: lock.unlink(missing_ok=True))
    print(f"AeroXAI preview on http://127.0.0.1:{args.port} · Syrus for guests: {'live, capped' if access.syrus_guests else 'off'} · lab: live · /twin and /optimization: not served")
    uvicorn.run("backend.web:app", host="127.0.0.1", port=args.port, workers=1, proxy_headers=True, forwarded_allow_ips="127.0.0.1", log_level="info")


if __name__ == "__main__":
    main()
