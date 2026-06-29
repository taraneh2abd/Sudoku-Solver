from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple synthetic empty-cell and digit samples.")
    parser.add_argument("--output-dir", default="data/synthetic/digits")
    parser.add_argument("--count-per-class", type=int, default=300)
    parser.add_argument("--font-scale", type=float, default=0.9)
    return parser.parse_args()


def make_digit(label: int, font_scale: float) -> np.ndarray:
    image = np.full((50, 50), 255, dtype=np.uint8)
    if label != 0:
        thickness = random.randint(1, 3)
        text = str(label)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = (50 - w) // 2 + random.randint(-3, 3)
        y = (50 + h) // 2 + random.randint(-3, 3)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, thickness, cv2.LINE_AA)

    angle = random.uniform(-10, 10)
    matrix = cv2.getRotationMatrix2D((25, 25), angle, random.uniform(0.9, 1.08))
    image = cv2.warpAffine(image, matrix, (50, 50), borderValue=255)
    noise = np.random.normal(0, random.uniform(2, 10), image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    for split in ["train", "val"]:
        for label in range(10):
            ensure_dir(root / split / str(label))

    for label in range(10):
        for index in range(args.count_per_class):
            split = "val" if index < max(20, args.count_per_class // 5) else "train"
            image = make_digit(label, args.font_scale)
            cv2.imwrite(str(root / split / str(label) / f"{label}_{index:05d}.png"), image)

    print(f"Synthetic dataset saved to {root}")


if __name__ == "__main__":
    main()
