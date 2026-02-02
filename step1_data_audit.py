import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ============ CONFIG ============
DATA_DIR = Path("data")
CLASSES = ["rock", "paper", "scissors"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
VAL_RATIO = 0.2

# Output split
OUT_DIR = Path("data_split")
TRAIN_DIR = OUT_DIR / "train"
VAL_DIR = OUT_DIR / "val"

# Sanity image
SANITY_PATH = Path("sanity_grid.png")
GRID_N = 3  # per class
TARGET_SIZE = (224, 224)  # just for visualization, not training yet

random.seed(SEED)


def list_images(folder: Path):
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def make_split():
    print("\n--- Checking dataset structure ---")
    for c in CLASSES:
        class_dir = DATA_DIR / c
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

    # Count images
    print("\n--- Class counts ---")
    all_files = {}
    total = 0
    for c in CLASSES:
        files = list_images(DATA_DIR / c)
        all_files[c] = files
        total += len(files)
        print(f"{c}: {len(files)}")

    if total == 0:
        raise RuntimeError("No images found. Check your data folder and extensions.")

    # Create split folders
    ensure_clean_dir(TRAIN_DIR)
    ensure_clean_dir(VAL_DIR)
    for c in CLASSES:
        (TRAIN_DIR / c).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / c).mkdir(parents=True, exist_ok=True)

    # Split per class (stratified)
    print("\n--- Creating train/val split ---")
    for c, files in all_files.items():
        files = files[:]  # copy
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_RATIO)) if len(files) > 1 else 0
        val_files = files[:n_val]
        train_files = files[n_val:]

        for src in train_files:
            dst = TRAIN_DIR / c / src.name
            shutil.copy2(src, dst)

        for src in val_files:
            dst = VAL_DIR / c / src.name
            shutil.copy2(src, dst)

        print(f"{c}: train={len(train_files)}, val={len(val_files)}")

    print(f"\n✅ Split created in: {OUT_DIR.resolve()}")


def save_sanity_grid():
    print("\n--- Creating sanity grid ---")
    tiles = []
    labels = []

    for c in CLASSES:
        files = list_images(DATA_DIR / c)
        if not files:
            continue
        pick = random.sample(files, k=min(GRID_N, len(files)))
        for p in pick:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, TARGET_SIZE)
            tiles.append(img)
            labels.append(c)

    if not tiles:
        raise RuntimeError("Could not read any images. Some files may be corrupted.")

    # Make a grid
    cols = GRID_N
    rows = int(np.ceil(len(tiles) / cols))
    h, w = TARGET_SIZE
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for i, img in enumerate(tiles):
        r = i // cols
        c = i % cols
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

    # Save
    cv2.imwrite(str(SANITY_PATH), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"✅ Sanity grid saved to: {SANITY_PATH.resolve()}")


if __name__ == "__main__":
    make_split()
    save_sanity_grid()
