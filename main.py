import sys
import cv2

from src.digit_recognizer import DigitRecognizer
from src.preprocess import preprocess
from src.grid_extraction import extract, GridNotFoundError
from src.cell_extraction import save_cells

from src.solve_sudoku import solve
import shutil
from pathlib import Path

OUTPUT_DIR = "results"

def prepare_output_dir(output_dir: str):
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

def print_board(board, title):

    print("\n" + title)
    print("-" * 30)

    for row in board:
        print(" ".join(str(x) for x in row))


def main():
    prepare_output_dir(OUTPUT_DIR)

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


# # main.py - نسخه کامل با تعریف LeNet
# import sys
# import json
# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from src.digit_recognizer import DigitRecognizer
# from src.preprocess import preprocess
# from src.grid_extraction import extract, GridNotFoundError
# from src.cell_extraction import save_cells
# from src.solve_sudoku import solve
# import shutil
# from pathlib import Path
# import os

# # ==================== تعریف کلاس LeNet در main.py ====================
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # ==================== بقیه کد main.py ====================
# OUTPUT_DIR = "results"

# def prepare_output_dir(output_dir: str):
#     output_dir = Path(output_dir)
#     if output_dir.exists():
#         shutil.rmtree(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

# def print_board(board, title, json_path=None):
#     print("\n" + title)
#     print("-" * 30)

#     for row in board:
#         print(" ".join(str(x) for x in row))

#     if json_path:
#         os.makedirs(os.path.dirname(json_path), exist_ok=True)

#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(board, f, indent=4)

# def main():
#     prepare_output_dir(OUTPUT_DIR)

#     if len(sys.argv) != 2:
#         print("Usage:")
#         print("python main.py image.jpg")
#         return

#     image_path = sys.argv[1]
#     image = cv2.imread(image_path)

#     if image is None:
#         sys.exit("Cannot read image")

#     print(f"Input : {image_path}")

#     pre = preprocess(image, OUTPUT_DIR)

#     try:
#         corners, warped, _ = extract(pre, image, OUTPUT_DIR)
#     except GridNotFoundError:
#         print("Grid not found")
#         return

#     cv2.imwrite(f"{OUTPUT_DIR}/final_warped.jpg", warped)

#     cells = save_cells(warped, OUTPUT_DIR)

#     recognizer = DigitRecognizer()
#     board = recognizer.predict_board(cells)

#     print_board(
#         board,
#         "Detected Sudoku",
#         "results/00_original.json"
#     )

#     solved = [row[:] for row in board]

#     if solve(solved):

#         print_board(
#             solved,
#             "Solved Sudoku",
#             "results/00_solved.json"
#         )

#     else:

#         print("\nNo solution found.")

#         with open("results/00_solved.json", "w", encoding="utf-8") as f:
#             json.dump("UNSOLVABLE", f)
# if __name__ == "__main__":
#     main()