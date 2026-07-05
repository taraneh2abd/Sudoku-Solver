"""
Sudoku grid detection pipeline
-------------------------------
Usage:
    python main.py <image_path> [--accuracy low|high] [--output output_dir]
"""

import sys
import argparse
import cv2

from src.preprocess import preprocess
from src.grid_extraction import extract


def main():
    parser = argparse.ArgumentParser(description="Sudoku grid extraction pipeline")
    parser.add_argument("image_path")
    parser.add_argument("--accuracy", choices=["low", "high"], default="low")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        sys.exit(f"ERROR: cannot read {args.image_path}")
    print(f"Input: {args.image_path}  ({image.shape[1]}x{image.shape[0]})  accuracy={args.accuracy}")

    pre_img = preprocess(image, args.output)
    result = extract(pre_img, image, args.output, accuracy=args.accuracy)

    corners, warped, _ = result
    if corners is None:
        print("\nFAILED: could not detect sudoku grid")
        sys.exit(1)

    cv2.imwrite(f"{args.output}/final_warped.jpg", warped)
    print(f"\nDone -> {args.output}/final_warped.jpg")


if __name__ == "__main__":
    main()
