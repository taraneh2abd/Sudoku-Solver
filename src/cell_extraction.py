from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import CELL_SIZE, EMPTY_CELL_FOREGROUND_RATIO, MODEL_IMAGE_SIZE
from src.utils import save_debug_image


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    raw: np.ndarray
    digit_image: np.ndarray
    is_empty: bool
    foreground_ratio: float


def remove_cell_border(cell: np.ndarray, margin_ratio: float = 0.14) -> np.ndarray:
    height, width = cell.shape[:2]
    margin_y = int(height * margin_ratio)
    margin_x = int(width * margin_ratio)
    return cell[margin_y : height - margin_y, margin_x : width - margin_x]


def normalize_digit(cell: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    cropped = remove_cell_border(gray)
    blurred = cv2.GaussianBlur(cropped, (3, 3), 0)
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(threshold)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 12:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    foreground_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if foreground_ratio < EMPTY_CELL_FOREGROUND_RATIO:
        return cv2.resize(mask, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)), foreground_ratio

    x, y, w, h = cv2.boundingRect(mask)
    digit = mask[y : y + h, x : x + w]
    side = max(w, h) + 8
    square = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - h) // 2
    x_offset = (side - w) // 2
    square[y_offset : y_offset + h, x_offset : x_offset + w] = digit
    normalized = cv2.resize(square, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return normalized, foreground_ratio


def extract_cells(warped_board: np.ndarray, debug_dir: Path | None = None) -> list[list[Cell]]:
    cells: list[list[Cell]] = []
    for row in range(9):
        cell_row: list[Cell] = []
        for col in range(9):
            y1, y2 = row * CELL_SIZE, (row + 1) * CELL_SIZE
            x1, x2 = col * CELL_SIZE, (col + 1) * CELL_SIZE
            raw = warped_board[y1:y2, x1:x2]
            digit_image, ratio = normalize_digit(raw)
            is_empty = ratio < EMPTY_CELL_FOREGROUND_RATIO
            cell = Cell(row=row, col=col, raw=raw, digit_image=digit_image, is_empty=is_empty, foreground_ratio=ratio)
            cell_row.append(cell)
            if debug_dir is not None:
                save_debug_image(debug_dir / f"cell_{row}_{col}.jpg", digit_image)
        cells.append(cell_row)
    return cells
