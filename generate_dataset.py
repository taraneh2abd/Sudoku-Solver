from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image

from src.cell_extraction import make_raw_cell, process_cell
from src.utils import ensure_dir

HF_DATASET = "Lexski/sudoku-image-recognition"

BOARD_SIZE = 450
CELL_SIZE = BOARD_SIZE // 9

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ImageFolder dataset from Lexski Sudoku dataset."
    )

    parser.add_argument(
        "--dataset",
        default=HF_DATASET,
    )

    parser.add_argument(
        "--output",
        default="data/processed/digits",
    )

    parser.add_argument(
        "--max-per-split",
        type=int,
        default=0,
        help="0 = all images",
    )

    return parser.parse_args()


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def warp_from_keypoints(
    image: np.ndarray,
    keypoints: list[float],
) -> np.ndarray:

    src = np.array(
        [
            [keypoints[0], keypoints[1]],
            [keypoints[6], keypoints[7]],
            [keypoints[4], keypoints[5]],
            [keypoints[2], keypoints[3]],
        ],
        dtype=np.float32,
    )

    dst = np.array(
        [
            [0, 0],
            [BOARD_SIZE - 1, 0],
            [BOARD_SIZE - 1, BOARD_SIZE - 1],
            [0, BOARD_SIZE - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        image,
        matrix,
        (BOARD_SIZE, BOARD_SIZE),
    )

    return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


def split_name(name: str) -> str:
    if name == "validation":
        return "val"
    return name


def cell_label(flags: list[int]) -> int | None:

    solved = int(flags[0]) == 1

    # خانه خالی
    if not solved:
        digits = [
            i
            for i, value in enumerate(flags[1:], start=1)
            if int(value) == 1
        ]

        if len(digits) == 0:
            return 0

        return None

    digits = [
        i
        for i, value in enumerate(flags[1:], start=1)
        if int(value) == 1
    ]

    if len(digits) != 1:
        return None

    return digits[0]


def prepare_dirs(root: Path):

    for split in ["train", "val", "test"]:
        for label in range(10):      # 0 تا 9
            ensure_dir(root / split / str(label))
            
def generate_split(dataset_split, split, output_dir, max_items):

    split = split_name(split)

    # counts = {str(i): 0 for i in range(1, 10)}
    counts = {str(i): 0 for i in range(10)}

    limit = len(dataset_split)

    if max_items > 0:
        limit = min(limit, max_items)

    for image_index in range(limit):

        item = dataset_split[image_index]

        image = pil_to_bgr(item["image"])

        warped = warp_from_keypoints(
            image,
            item["keypoints"],
        )

        for row in range(9):

            for col in range(9):

                label = cell_label(
                    item["cells"][row][col]
                )

                if label is None:
                    continue

                cell = make_raw_cell(
                    warped,
                    row,
                    col,
                )

                processed, _kept = process_cell(
                    cell.image
                )

                filename = (
                    f"{image_index:06d}"
                    f"_r{row}"
                    f"_c{col}.png"
                )

                cv2.imwrite(
                    str(
                        output_dir
                        / split
                        / str(label)
                        / filename
                    ),
                    processed,
                )

                counts[str(label)] += 1

    return counts


def main():

    args = parse_args()

    output_dir = Path(args.output)

    prepare_dirs(output_dir)

    dataset = load_dataset(args.dataset)

    summary = {}

    for split in dataset.keys():

        print(f"Generating {split}")

        summary[split_name(split)] = generate_split(
            dataset[split],
            split,
            output_dir,
            args.max_per_split,
        )

    print()

    print("Dataset generated.")

    print(output_dir)

    print()

    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()