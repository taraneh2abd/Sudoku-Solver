from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.cell_extraction import extract_cells
from src.image_pipeline import detect_and_warp_board
from src.overlay import overlay_solution
from src.predict import DigitPredictor
from src.solver import solve_sudoku, validate_grid
from src.utils import ensure_dir, save_debug_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect, recognize, and solve Sudoku from an image.")
    parser.add_argument("--image", required=True, help="Path to input Sudoku image.")
    parser.add_argument("--model", default="models/digit_cnn.pt", help="Path to trained CNN weights.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output images.")
    parser.add_argument("--debug", action="store_true", help="Save intermediate processing images.")
    parser.add_argument("--device", default="cpu", help="Torch device: cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = ensure_dir(args.output_dir)
    debug_dir = ensure_dir(output_dir / "debug_steps")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    board = detect_and_warp_board(image, debug_dir=debug_dir if args.debug else None)
    print(board)
    # cells = extract_cells(board.warped, debug_dir=debug_dir / "cells" if args.debug else None)

    # predictor = DigitPredictor(args.model, device=args.device)
    # grid = np.zeros((9, 9), dtype=np.int32)

    # for row in range(9):
    #     for col in range(9):
    #         cell = cells[row][col]
    #         grid[row, col] = 0 if cell.is_empty else predictor.predict_digit(cell.digit_image)

    # print("Recognized grid:")
    # print(grid)

    # if not validate_grid(grid.tolist()):
    #     raise ValueError("Recognized Sudoku grid is invalid before solving. Check digit recognition output.")

    # solved = [list(map(int, row)) for row in grid.tolist()]
    # if not solve_sudoku(solved):
    #     raise ValueError("Sudoku has no valid solution.")

    # solved_array = np.array(solved, dtype=np.int32)
    # print("Solved grid:")
    # print(solved_array)

    # solved_path = output_dir / "solved_overlay.jpg"
    # overlay = overlay_solution(image, board.corners, grid, solved_array)
    # cv2.imwrite(str(solved_path), overlay)

    # warped_path = output_dir / "warped_grid.jpg"
    # save_debug_image(warped_path, board.warped)
    # print(f"Saved warped grid: {warped_path}")
    # print(f"Saved solved overlay: {solved_path}")


if __name__ == "__main__":
    main()
