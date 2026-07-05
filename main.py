"""
Usage:
python main.py C:\\Users\\T.Abdellahi\\Desktop\\term8\\vision\\proj\\FINAL\\Sudoku-Solver\\data\\test\\00000.jpg accuracy low--output results
"""
import sys
import cv2

from src.preprocess import preprocess
from src.grid_extraction import extract, GridNotFoundError
from src.cell_extraction import save_cells

from src.digit_model_rec import DigitRecognizer
from src.solve_sudoku import solve


OUTPUT_DIR = "results"


def print_board(board, title):

    print("\n" + title)
    print("-" * 30)

    for row in board:
        print(" ".join(str(x) for x in row))


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print("python main.py image.jpg")
        return

    image_path = sys.argv[1]

    image = cv2.imread(image_path)

    if image is None:
        sys.exit("Cannot read image")

    print(f"Input : {image_path}")

    pre = preprocess(image, OUTPUT_DIR)

    try:
        corners, warped, _ = extract(pre, image, OUTPUT_DIR)

    except GridNotFoundError:

        print("Grid not found")
        return

    cv2.imwrite(f"{OUTPUT_DIR}/final_warped.jpg", warped)

    cells = save_cells(warped, OUTPUT_DIR)

    recognizer = DigitRecognizer()

    board = recognizer.predict_board(cells)

    print_board(board, "Detected Sudoku")

    solved = [row[:] for row in board]

    if solve(solved):

        print_board(solved, "Solved Sudoku")

    else:

        print("\nNo solution found.")


if __name__ == "__main__":
    main()