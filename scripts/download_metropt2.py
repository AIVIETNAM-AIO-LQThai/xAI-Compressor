from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from ml.data.metropt2 import file_digests

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "metropt2.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    return payload["dataset"]


def _download(*, url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    partial = target.with_suffix(target.suffix + ".part")
    existing_bytes = (
        partial.stat().st_size if partial.exists() else 0
    )

    headers = {
        "User-Agent": "AeroXAI-MetroPT2-research/1.0"
    }

    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    request = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", None)
        resumed = existing_bytes > 0 and status == 206
        mode = "ab" if resumed else "wb"

        if existing_bytes > 0 and not resumed:
            print(
                "Server did not honor Range; restarting download."
            )

        written = existing_bytes if resumed else 0

        with partial.open(mode) as destination:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break

                destination.write(chunk)
                previous = written
                written += len(chunk)

                if (
                    written // (256 * 1024 * 1024)
                    != previous // (256 * 1024 * 1024)
                ):
                    print(
                        "Downloaded "
                        f"{written / 1024**3:.2f} GiB"
                    )

    shutil.move(str(partial), str(target))


def main() -> None:
    config = load_config()
    target = ROOT / config["raw_csv"]
    expected_md5 = str(config["expected_md5"]).lower()

    if target.exists():
        existing = file_digests(target)

        if existing["md5"] == expected_md5:
            print(
                "MetroPT2 already exists and matches "
                "the published MD5."
            )
            print(f"SHA-256: {existing['sha256']}")
            return

        raise RuntimeError(
            "Existing MetroPT2 file does not match the "
            "published MD5.\n"
            f"Path: {target}\n"
            f"Expected MD5: {expected_md5}\n"
            f"Actual MD5:   {existing['md5']}\n"
            "Move or delete the file before retrying."
        )

    print(
        "Downloading MetroPT2 from the official Zenodo record."
    )
    print(f"Target: {target}")

    _download(
        url=str(config["source_url"]),
        target=target,
    )

    digests = file_digests(target)

    if digests["md5"] != expected_md5:
        raise RuntimeError(
            "Downloaded MetroPT2 file failed published "
            "MD5 verification.\n"
            f"Expected: {expected_md5}\n"
            f"Actual:   {digests['md5']}"
        )

    print("Published MD5 verified.")
    print(f"MD5:     {digests['md5']}")
    print(f"SHA-256: {digests['sha256']}")


if __name__ == "__main__":
    main()
