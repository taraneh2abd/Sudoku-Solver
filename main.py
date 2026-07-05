"""
Usage:
python main.py C:\\Users\\T.Abdellahi\\Desktop\\term8\\vision\\proj\\FINAL\\Sudoku-Solver\\data\\test\\00000.jpg accuracy low--output results
"""
import sys
import argparse
import cv2

from src.preprocess import preprocess
from src.grid_extraction import extract, GridNotFoundError
from src.cell_extraction import save_cells


def main():
    parser = argparse.ArgumentParser(description="Sudoku grid extraction pipeline")
    parser.add_argument("image_path")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        sys.exit(f"ERROR: cannot read {args.image_path}")
    print(f"Input: {args.image_path}  ({image.shape[1]}x{image.shape[0]})")

    pre_img = preprocess(image, args.output)

    try:
        result = extract(pre_img, image, args.output)
    except GridNotFoundError:
        result = (None, None, None)

    corners, warped, _ = result
    if corners is None:
        print("\nFAILED: could not detect sudoku grid")
        sys.exit(1)

    cv2.imwrite(f"{args.output}/final_warped.jpg", warped)
    cv2.imwrite(f"{args.output}/final_warped.jpg", warped)

    save_cells(warped, args.output)

    print(f"\nDone -> {args.output}/final_warped.jpg")
    print(f"\nDone -> {args.output}/final_warped.jpg")


if __name__ == "__main__":
    main()
