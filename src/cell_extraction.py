from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .grid_extraction import WARP_SIZE

CELL_SIZE = WARP_SIZE // 9
DIGIT_SIZE = 224

MIN_DIGIT_AREA_RATIO = 0.015
MIN_DIGIT_HEIGHT_RATIO = 0.25
CELL_MARGIN_RATIO = 0.12


@dataclass
class Cell:
    image: np.ndarray
    is_empty: bool
    ink_ratio: float


def save_cells(warped, output_dir):
    """
    warped: the normalized/warped grayscale grid image returned by
            src.grid_extraction.extract() (already preprocessed).
    Splits it into the 81 cells, saves the debug stages + montage + the
    individual non-empty digit crops under output_dir, and returns the
    list of Cell objects (row-major).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # one problem we had was the hole for 6/8/9 getting filled in by the median blur,
    # so we use a more permissive threshold to get the grid lines,
    #  and a more strict one to get the digit mask for cell extraction
    warped_blurred = cv2.medianBlur(warped, 3)

    # keeps grid lines thick enough for boundary detection
    warped_binary = cv2.adaptiveThreshold(
        warped_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10,
    )
    # C=15 removes noise pixels
    # line geometry from warped_binary is applied to it
    warped_binary_clean = cv2.adaptiveThreshold(
        warped_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15,
    )

    line_free, horizontal_lines, vertical_lines = _remove_grid_lines(warped_binary, warped_binary_clean)
    row_bounds = _grid_boundaries(horizontal_lines, axis=1)
    col_bounds = _grid_boundaries(vertical_lines, axis=0)

    cv2.imwrite(str(out_dir / "07_warped_binary.png"), warped_binary)
    cv2.imwrite(str(out_dir / "08_lines_removed.png"), line_free)

    cell_grid = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    for bound in row_bounds:
        cv2.line(cell_grid, (0, int(bound)), (WARP_SIZE, int(bound)), (0, 255, 0), 1)
    for bound in col_bounds:
        cv2.line(cell_grid, (int(bound), 0), (int(bound), WARP_SIZE), (0, 255, 0), 1)
    cv2.imwrite(str(out_dir / "09_cell_grid.png"), cell_grid)

    cells = [
        _extract_cell(line_free, warped, row_bounds, col_bounds, row, col)
        for row in range(9)
        for col in range(9)
    ]

    cv2.imwrite(str(out_dir / "10_cells_montage.png"), cell_montage(cells))
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    for index, cell in enumerate(cells):
        if not cell.is_empty:
            row, col = divmod(index, 9)
            cv2.imwrite(str(cells_dir / f"cell_r{row}c{col}.png"), cell.image)

    filled = sum(not cell.is_empty for cell in cells)
    print(f"grid found; {filled} filled cells, {81 - filled} empty")
    print(_empty_mask_text(cells))

    return cells


def _remove_grid_lines(warped_binary, digit_binary=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CELL_SIZE, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, CELL_SIZE))
    horizontal = cv2.morphologyEx(warped_binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(warped_binary, cv2.MORPH_OPEN, vertical_kernel)
    lines = cv2.dilate(
        cv2.bitwise_or(horizontal, vertical),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    target = digit_binary if digit_binary is not None else warped_binary
    line_free = cv2.bitwise_and(target, cv2.bitwise_not(lines))
    return line_free, horizontal, vertical


def _grid_boundaries(lines_mask, axis) -> np.ndarray:
    profile = (lines_mask > 0).sum(axis=axis)
    search = CELL_SIZE // 3
    bounds = np.empty(10, dtype=int)

    for index in range(10):
        expected = min(index * CELL_SIZE, WARP_SIZE - 1)
        low = max(0, expected - search)
        window = profile[low : min(WARP_SIZE, expected + search + 1)]

        if window.max() >= 0.3 * WARP_SIZE:
            bounds[index] = low + int(window.argmax())
        else:
            bounds[index] = expected

    return np.maximum.accumulate(bounds)


def _extract_cell(line_free, warped, row_bounds, col_bounds, row, col) -> Cell:

    y0, y1 = row_bounds[row], row_bounds[row + 1]
    x0, x1 = col_bounds[col], col_bounds[col + 1]
    y0, y1 = y0 + int((y1 - y0) * CELL_MARGIN_RATIO), y1 - int((y1 - y0) * CELL_MARGIN_RATIO)
    x0, x1 = x0 + int((x1 - x0) * CELL_MARGIN_RATIO), x1 - int((x1 - x0) * CELL_MARGIN_RATIO)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return Cell(np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.uint8), True, 0.0)

    interior = line_free[y0:y1, x0:x1]
    ink_ratio = float(cv2.countNonZero(interior)) / interior.size

    digit_mask = _find_digit_mask(interior)
    if digit_mask is None:
        return Cell(np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.uint8), True, ink_ratio)

    # crop the digit from the warped *grayscale*: a binary mask fills the
    # holes of 6/8/9 under noise, grayscale keeps them visible for Phase 2
    ys, xs = np.nonzero(digit_mask)
    pad = 3
    gy0, gy1 = max(0, y0 + ys.min() - pad), min(WARP_SIZE, y0 + ys.max() + 1 + pad)
    gx0, gx1 = max(0, x0 + xs.min() - pad), min(WARP_SIZE, x0 + xs.max() + 1 + pad)
    mask_full = np.zeros(line_free.shape, np.uint8)
    mask_full[y0:y1, x0:x1] = digit_mask
    digit = _normalize_digit(warped[gy0:gy1, gx0:gx1], mask_full[gy0:gy1, gx0:gx1])
    return Cell(digit, False, ink_ratio)


def _find_digit_mask(interior) -> np.ndarray | None:
    height, width = interior.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(interior, connectivity=8)

    anchor, anchor_area = None, 0
    for label in range(1, count):
        x, y, w, h, area = stats[label]

        # size checks
        if area < MIN_DIGIT_AREA_RATIO * interior.size:
            continue
        if h < MIN_DIGIT_HEIGHT_RATIO * height:
            continue

        # aspect Ratio: Digits are generally taller than they are wide.
        if w > 1.2 * h:
            continue

        # fill Density: Wispy smudges/wrinkles have large bounding boxes but very few actual pixels. Digits are solid strokes.
        density = area / (w * h)
        if density < 0.15:
            continue

        # centroid check
        cx, cy = centroids[label]
        if not (0.15 * width < cx < 0.85 * width and 0.15 * height < cy < 0.85 * height):
            continue

        if area > anchor_area:
            anchor, anchor_area = label, area

    if anchor is None:
        return None

    ax, ay, aw, ah, _ = stats[anchor]
    grow = max(2, min(height, width) // 8)
    left, top = ax - grow, ay - grow
    right, bottom = ax + aw + grow, ay + ah + grow
    min_fragment = 0.3 * MIN_DIGIT_AREA_RATIO * interior.size

    mask = np.zeros_like(interior)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if label != anchor:
            if area < min_fragment:
                continue
            if x + w < left or y + h < top or x > right or y > bottom:
                continue
        mask[labels == label] = 255

    return mask


def _normalize_digit(gray_crop, mask_crop) -> np.ndarray:
    # 1. Closes the mask with a scale aware kernel (~4% of the smaller side)
    #    to bridge small gaps without thickening strokes or filling loops.
    # 2. Inverts the grayscale (white digit on black background).
    # 3. Masks out everything outside the digit.
    # 4. Normalizes pixel values to 0–255.
    # 5. Pads to a square canvas at 1.3× the digit size.
    # 6. Resize

    h, w = mask_crop.shape
    k = max(1, round(min(h, w) * 0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    mask = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, kernel)

    inverted = cv2.bitwise_not(gray_crop)
    digit = cv2.bitwise_and(inverted, inverted, mask=mask)
    digit = cv2.normalize(digit, None, 0, 255, cv2.NORM_MINMAX)
    height, width = digit.shape
    side = int(max(height, width) * 1.3)
    canvas = np.zeros((side, side), np.uint8)
    y0 = (side - height) // 2
    x0 = (side - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = digit
    return cv2.resize(canvas, (DIGIT_SIZE, DIGIT_SIZE), interpolation=cv2.INTER_AREA)


def cell_montage(cells) -> np.ndarray:
    pad, tile = 3, DIGIT_SIZE
    step = tile + 2 * pad
    canvas = np.full((9 * step, 9 * step, 3), 40, np.uint8)
    for index, cell in enumerate(cells):
        row, col = divmod(index, 9)
        y0, x0 = row * step, col * step
        frame_color = (40, 40, 40) if cell.is_empty else (0, 160, 0)
        cv2.rectangle(canvas, (x0, y0), (x0 + step - 1, y0 + step - 1), frame_color, pad)
        patch = cv2.cvtColor(cell.image, cv2.COLOR_GRAY2BGR)
        canvas[y0 + pad : y0 + pad + tile, x0 + pad : x0 + pad + tile] = patch
    return canvas


def _empty_mask_text(cells) -> str:
    #  '#' = digit , '.' = empty
    rows = [cells[r * 9 : r * 9 + 9] for r in range(9)]
    return "\n".join(
        " ".join("." if cell.is_empty else "#" for cell in row)
        for row in rows
    )
