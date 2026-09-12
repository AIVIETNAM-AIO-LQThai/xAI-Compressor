from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "metropt.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["dataset"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary = target.with_suffix(
        target.suffix + ".part"
    )

    print(f"Downloading:\n  {url}")
    print(f"To:\n  {target}")

    urllib.request.urlretrieve(url, temporary)

    temporary.replace(target)


def extract_named_file(
    archive: zipfile.ZipFile,
    filename: str,
    target: Path,
) -> None:
    matches = [
        member
        for member in archive.namelist()
        if Path(member).name == filename
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {filename!r}; "
            f"found {matches}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)

    with archive.open(matches[0]) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination)


def main() -> None:
    config = load_config()

    zip_path = ROOT / config["raw_zip"]
    csv_path = ROOT / config["raw_csv"]
    pdf_path = ROOT / config["raw_description_pdf"]

    if not zip_path.exists():
        download_file(
            config["source_url"],
            zip_path,
        )

    actual_hash = sha256_file(zip_path)
    expected_hash = config["expected_zip_sha256"]

    print(f"SHA-256: {actual_hash}")

    if actual_hash != expected_hash:
        raise RuntimeError(
            "MetroPT-3 archive checksum mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    print("Checksum verified.")

    with zipfile.ZipFile(zip_path) as archive:
        extract_named_file(
            archive,
            "MetroPT3(AirCompressor).csv",
            csv_path,
        )

        extract_named_file(
            archive,
            "Data Description_Metro.pdf",
            pdf_path,
        )

    print(f"CSV extracted: {csv_path}")
    print(f"PDF extracted: {pdf_path}")


if __name__ == "__main__":
    main()