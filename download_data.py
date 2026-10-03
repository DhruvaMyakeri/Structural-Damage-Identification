"""Download and unpack the concrete crack image collection.

The assigned paper used PEER Hub ImageNet. That collection is sent only after
an email request on the PEER site, so this project uses a public stand-in with
the same kind of label: a close-up is either cracked (damaged) or not.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
ZIP_DIR = ROOT / "data" / "zips"
RAW_DIR = ROOT / "data" / "raw"

BASE = (
    "https://huggingface.co/datasets/mohammadnajeeb/"
    "concrete_crack_images/resolve/main/data/"
)
ZIPS = ("train.zip", "validation.zip", "test.zip")


def _has_images(folder: Path) -> bool:
    if not folder.exists():
        return False
    for path in folder.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return True
    return False


def ensure_data(extract: bool = False) -> Path:
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name in ZIPS:
        destination = ZIP_DIR / name
        if destination.exists() and destination.stat().st_size > 1_000_000:
            print(f"zip already present: {name}")
            continue
        url = BASE + name
        print(f"downloading {url}")
        urlretrieve(url, destination)
        print(f"saved {destination} ({destination.stat().st_size} bytes)")

    if not extract:
        return ZIP_DIR

    if _has_images(RAW_DIR / "train") and _has_images(RAW_DIR / "validation") and _has_images(RAW_DIR / "test"):
        print(f"images already unpacked in {RAW_DIR}")
        return RAW_DIR

    for name in ZIPS:
        print(f"extracting {name}")
        with zipfile.ZipFile(ZIP_DIR / name) as archive:
            archive.extractall(RAW_DIR)
    return RAW_DIR


if __name__ == "__main__":
    print(ensure_data())
