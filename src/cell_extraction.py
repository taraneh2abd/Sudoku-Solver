from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .grid_extraction import WARP_SIZE

# CELL_SIZE = 450 // 9

CELL_SIZE = WARP_SIZE // 9
CELL_MARGIN_RATIO = 0.12

WHITE_RATIO_FOR_INVERSE = 0.55
MIN_NOISE_AREA = 15

EDGE_RATIO = 0.0


@dataclass
class Cell:
    image: np.ndarray
    is_empty: bool


def save_cells(warped, output_dir):
    """
    warped: تصویر خاکستری وارپ‌شده‌ی گرید.

    ترتیب دقیق کار:
      1) ابتدا هر ۸۱ خونه از تصویر خام برش زده و در cells ریخته می‌شود.
         خونه‌های خالی همینجا مشخص و بدون پردازش رها می‌شوند.
      2) هر خونه‌ی غیرخالی باینری (۰ و ۲۵۵) می‌شود.
      3) اگر بیش از ۵۵٪ پیکسل‌های آن سفید باشند، اینورس می‌شود.
      4) فقط خطوط افقی و عمودی نزدیک لبه‌های سلول حذف می‌شوند.
      5) فقط کامپوننت‌های متصل بزرگ نگه داشته شده و نویز حذف می‌شود.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = [
        make_raw_cell(warped, row, col)
        for row in range(9)
        for col in range(9)
    ]

    for cell in cells:
        if not cell.is_empty:
            cell.image = process_cell(cell.image)

    cells_dir = out_dir / "cells"
    cells_dir.mkdir(exist_ok=True)

    for index, cell in enumerate(cells):
        if not cell.is_empty:
            row, col = divmod(index, 9)
            cv2.imwrite(str(cells_dir / f"cell_r{row}c{col}.png"), cell.image)

    filled = sum(not cell.is_empty for cell in cells)
    print(f"grid found; {filled} filled cells, {81 - filled} empty")

    return cells


def make_raw_cell(warped, row, col) -> Cell:
    """خونه را از تصویر خاکستری خام برش می‌زند و خالی/پر بودنش را با یک چک ساده تشخیص می‌دهد."""

    y0, y1 = row * CELL_SIZE, (row + 1) * CELL_SIZE
    x0, x1 = col * CELL_SIZE, (col + 1) * CELL_SIZE

    y0 = y0 + int((y1 - y0) * CELL_MARGIN_RATIO)
    y1 = y1 - int((y1 - y0) * CELL_MARGIN_RATIO)

    x0 = x0 + int((x1 - x0) * CELL_MARGIN_RATIO)
    x1 = x1 - int((x1 - x0) * CELL_MARGIN_RATIO)

    crop = warped[y0:y1, x0:x1]

    is_empty = float(np.std(crop)) < 8.0

    return Cell(crop, is_empty)


def process_cell(crop) -> np.ndarray:
    _, binary = cv2.threshold(
        crop,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    white_ratio = cv2.countNonZero(binary) / binary.size
    if white_ratio > WHITE_RATIO_FOR_INVERSE:
        binary = cv2.bitwise_not(binary)

    h, w = binary.shape

    # ماسک خطوطی که باید حذف شوند
    remove_mask = np.zeros_like(binary)

    # پیدا کردن خطوط
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=min(h, w) // 3,
        maxLineGap=3,
    )

    if lines is not None:
        margin_x = int(w * EDGE_RATIO)
        margin_y = int(h * EDGE_RATIO)

        for line in lines:
            x1, y1, x2, y2 = line[0]

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dx <= 2:
                x = (x1 + x2) // 2
                if x <= margin_x or x >= w - margin_x:
                    cv2.line(remove_mask, (x1, y1), (x2, y2), 255, 3)

            elif dy <= 2:
                y = (y1 + y2) // 2
                if y <= margin_y or y >= h - margin_y:
                    cv2.line(remove_mask, (x1, y1), (x2, y2), 255, 3)

    binary = cv2.bitwise_and(binary, cv2.bitwise_not(remove_mask))

    return clean_noise(binary)


def clean_noise(cell_image) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cell_image,
        connectivity=8,
    )

    cleaned = np.zeros_like(cell_image)

    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= MIN_NOISE_AREA:
            cleaned[labels == label] = 255

    return cleaned


# img = cv2.imread("tests/cell.png", cv2.IMREAD_GRAYSCALE)

# result = _process_cell(img)

# cv2.imwrite("tests/result.png", result)