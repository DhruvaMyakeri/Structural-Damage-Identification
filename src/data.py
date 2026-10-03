"""Find images, build the splits, and turn pictures into model inputs.

Labels follow the paper: 0 is undamaged, 1 is damaged. In this collection the
folder Positive holds cracked concrete and Negative holds intact concrete.

Photos are read from the downloaded zip files. Unpacking forty thousand small
files on Windows is much slower than reading the archives in place.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ZIP_DIR = ROOT / "data" / "zips"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_ZIPS = {
    "train": "train.zip",
    "validation": "validation.zip",
    "test": "test.zip",
}

_OPEN_ZIPS: dict[Path, zipfile.ZipFile] = {}


def _class_from_folder(name: str) -> int | None:
    lowered = name.lower()
    if lowered in {"negative", "undamaged", "no_crack", "nocrack"}:
        return 0
    if lowered in {"positive", "damaged", "crack"}:
        return 1
    return None


def open_image(ref) -> Image.Image:
    """Open a filesystem path or a (zip path, member name) pair."""
    if isinstance(ref, tuple):
        zip_path, member = ref
        zip_path = Path(zip_path)
        archive = _OPEN_ZIPS.get(zip_path)
        if archive is None:
            archive = zipfile.ZipFile(zip_path)
            _OPEN_ZIPS[zip_path] = archive
        raw = archive.read(member)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(ref).convert("RGB")


def image_name(ref) -> str:
    if isinstance(ref, tuple):
        return Path(ref[1]).name
    return Path(ref).name


def list_split_zip(split_name: str, zip_dir: Path = ZIP_DIR) -> tuple[list[tuple[Path, str]], np.ndarray]:
    zip_path = zip_dir / SPLIT_ZIPS[split_name]
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}. Run python download_data.py first.")
    paths: list[tuple[Path, str]] = []
    labels: list[int] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            suffix = Path(member).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            label = _class_from_folder(Path(member).parent.name)
            if label is None:
                continue
            paths.append((zip_path, member))
            labels.append(label)
    if not paths:
        raise FileNotFoundError(f"No images inside {zip_path}")
    order = np.argsort([item[1] for item in paths])
    paths = [paths[i] for i in order]
    return paths, np.asarray(labels, dtype=np.int64)[order]


def load_splits(zip_dir: Path = ZIP_DIR) -> dict[str, tuple[list, np.ndarray]]:
    return {name: list_split_zip(name, zip_dir) for name in SPLIT_ZIPS}


def count_classes(labels: np.ndarray) -> dict[str, int]:
    return {
        "undamaged": int((labels == 0).sum()),
        "damaged": int((labels == 1).sum()),
        "total": int(labels.shape[0]),
    }


def stratified_take(
    paths: list[Path],
    labels: np.ndarray,
    per_class: int | None,
    seed: int,
) -> tuple[list[Path], np.ndarray]:
    """Keep up to `per_class` images from each class. None keeps every image."""
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for cls in (0, 1):
        index = np.flatnonzero(labels == cls)
        rng.shuffle(index)
        if per_class is not None:
            index = index[:per_class]
        chosen.extend(index.tolist())
    order = rng.permutation(len(chosen))
    chosen = [chosen[i] for i in order]
    subset_paths = [paths[i] for i in chosen]
    subset_labels = labels[np.asarray(chosen, dtype=np.int64)]
    return subset_paths, subset_labels


def pixels_to_vectors(paths: list, size: int = 32) -> np.ndarray:
    """Resize, flatten, and map 8-bit pixels into [-1, 1).

    The paper's formula is x = (pixel / 128) - 1. Full 224x224x3 vectors are
    150,528 numbers long, which makes a kernel SVM impractical here, so the
    pictures are reduced to `size` by `size` first. The paper's word "scaled"
    covers both this resize and the [-1, 1) map.
    """
    width = size * size * 3
    matrix = np.empty((len(paths), width), dtype=np.float32)
    for row, path in enumerate(paths):
        image = open_image(path).resize((size, size), Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32).reshape(-1)
        matrix[row] = (pixels / 128.0) - 1.0
        if row and row % 2000 == 0:
            print(f"  vectorised {row}/{len(paths)}")
    return matrix
