"""Download the concrete crack image collection.

The assigned paper used PEER Hub ImageNet. That collection is sent only after
an email request on the PEER site, so this project uses a public stand-in with
the same kind of label: a close-up is either cracked (damaged) or not.

Training reads the zip members in place. The archives are not unpacked.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
ZIP_DIR = ROOT / "data" / "zips"

BASE = (
    "https://huggingface.co/datasets/mohammadnajeeb/"
    "concrete_crack_images/resolve/main/data/"
)
ZIPS = ("train.zip", "validation.zip", "test.zip")


def ensure_data() -> Path:
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    for name in ZIPS:
        destination = ZIP_DIR / name
        if destination.exists() and destination.stat().st_size > 1_000_000:
            print(f"zip already present: {name}")
            continue
        url = BASE + name
        print(f"downloading {url}")
        urlretrieve(url, destination)
        print(f"saved {destination} ({destination.stat().st_size} bytes)")

    return ZIP_DIR


if __name__ == "__main__":
    print(ensure_data())
