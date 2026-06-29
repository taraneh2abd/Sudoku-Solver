from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image

from src.config import BOARD_SIZE, CELL_SIZE
from src.cell_extraction import normalize_digit
from src.utils import ensure_dir


HF_DATASET = "Lexski/sudoku-image-recognition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare the Lexski Sudoku image dataset from HuggingFace."
    )
    parser.add_argument("--dataset", default=HF_DATASET, help="HuggingFace dataset id.")
    parser.add_argument("--raw-dir", default="data/raw/lexski", help="Where full Sudoku images are saved.")
    parser.add_argument("--digits-dir", default="data/processed/digits", help="ImageFolder output for CNN training.")
    parser.add_argument("--metadata-dir", default="data/processed/lexski_metadata", help="Where labels/keypoints are saved.")
    parser.add_argument("--max-per-split", type=int, default=0, help="Optional limit for quick tests. 0 means all.")
    parser.add_argument("--save-empty", action="store_true", help="Also save reliable blank cells as class 0.")
    parser.add_argument("--empty-train-count", type=int, default=1200, help="Synthetic empty cells added to train/0.")
    parser.add_argument("--empty-val-count", type=int, default=250, help="Synthetic empty cells added to val/0.")
    return parser.parse_args()


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def warp_from_keypoints(image: np.ndarray, keypoints: list[float]) -> np.ndarray:
    # Dataset order is top-left, bottom-left, bottom-right, top-right.
    source = np.array(
        [
            [keypoints[0], keypoints[1]],
            [keypoints[6], keypoints[7]],
            [keypoints[4], keypoints[5]],
            [keypoints[2], keypoints[3]],
        ],
        dtype="float32",
    )
    destination = np.array(
        [
            [0, 0],
            [BOARD_SIZE - 1, 0],
            [BOARD_SIZE - 1, BOARD_SIZE - 1],
            [0, BOARD_SIZE - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, transform, (BOARD_SIZE, BOARD_SIZE))


def split_name(name: str) -> str:
    return "val" if name == "validation" else name


def cell_label(flags: list[int], save_empty: bool) -> int | None:
    solved = int(flags[0]) == 1
    digits = [index for index, value in enumerate(flags[1:], start=1) if int(value) == 1]
    if solved and len(digits) == 1:
        return digits[0]
    if save_empty and not solved and len(digits) == 0:
        return 0
    return None


def prepare_split(dataset_split, split: str, raw_dir: Path, digits_dir: Path, metadata_dir: Path, max_items: int, save_empty: bool) -> dict[str, int]:
    out_split = split_name(split)
    counts = {str(label): 0 for label in range(10)}
    raw_split_dir = ensure_dir(raw_dir / out_split)
    meta_split_dir = ensure_dir(metadata_dir / out_split)

    limit = len(dataset_split) if max_items <= 0 else min(max_items, len(dataset_split))
    for index in range(limit):
        item = dataset_split[index]
        image = pil_to_bgr(item["image"])
        raw_path = raw_split_dir / f"{index:05d}.jpg"
        cv2.imwrite(str(raw_path), image)

        metadata = {
            "source_index": index,
            "raw_image": str(raw_path.as_posix()),
            "cells": item["cells"],
            "keypoints": item["keypoints"],
        }
        (meta_split_dir / f"{index:05d}.json").write_text(json.dumps(metadata), encoding="utf-8")

        warped = warp_from_keypoints(image, item["keypoints"])
        for row in range(9):
            for col in range(9):
                label = cell_label(item["cells"][row][col], save_empty=save_empty)
                if label is None:
                    continue
                y1, y2 = row * CELL_SIZE, (row + 1) * CELL_SIZE
                x1, x2 = col * CELL_SIZE, (col + 1) * CELL_SIZE
                normalized, _ = normalize_digit(warped[y1:y2, x1:x2])
                label_dir = ensure_dir(digits_dir / out_split / str(label))
                filename = f"{index:05d}_{row}_{col}.png"
                cv2.imwrite(str(label_dir / filename), normalized)
                counts[str(label)] += 1

    return counts


def ensure_imagefolder_dirs(digits_dir: Path) -> None:
    for split in ["train", "val", "test"]:
        for label in range(10):
            ensure_dir(digits_dir / split / str(label))


def make_empty_cell() -> np.ndarray:
    base = np.full((28, 28), np.random.randint(230, 256), dtype=np.uint8)
    noise = np.random.normal(0, np.random.uniform(2, 9), base.shape).astype(np.int16)
    image = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if np.random.random() < 0.35:
        x1 = np.random.randint(0, 24)
        y1 = np.random.randint(0, 24)
        x2 = min(27, x1 + np.random.randint(2, 8))
        y2 = min(27, y1 + np.random.randint(2, 8))
        cv2.rectangle(image, (x1, y1), (x2, y2), int(np.random.randint(205, 245)), -1)
    return 255 - image


def add_synthetic_empty_cells(digits_dir: Path, train_count: int, val_count: int) -> dict[str, int]:
    counts = {"train": train_count, "val": val_count}
    for split, count in counts.items():
        label_dir = ensure_dir(digits_dir / split / "0")
        for index in range(count):
            image = make_empty_cell()
            cv2.imwrite(str(label_dir / f"empty_{index:05d}.png"), image)
    return counts


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    digits_dir = Path(args.digits_dir)
    metadata_dir = Path(args.metadata_dir)
    ensure_imagefolder_dirs(digits_dir)
    ensure_dir(raw_dir)
    ensure_dir(metadata_dir)

    dataset = load_dataset(args.dataset)
    summary: dict[str, dict[str, int]] = {}
    for split in dataset.keys():
        print(f"Preparing split: {split}")
        summary[split_name(split)] = prepare_split(
            dataset[split],
            split,
            raw_dir,
            digits_dir,
            metadata_dir,
            args.max_per_split,
            args.save_empty,
        )

    empty_counts = add_synthetic_empty_cells(digits_dir, args.empty_train_count, args.empty_val_count)
    summary["synthetic_empty"] = {"0": empty_counts["train"] + empty_counts["val"]}

    summary_path = metadata_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Raw Sudoku images: {raw_dir}")
    print(f"Digit ImageFolder dataset: {digits_dir}")
    print(f"Summary: {summary_path}")
    print("Next: python -m src.train --data-dir data/processed/digits --epochs 15 --output models/digit_cnn.pt")


if __name__ == "__main__":
    main()
