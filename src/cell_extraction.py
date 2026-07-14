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


def save_cells(warped, warped_original, output_dir):
    """
    warped: تصویر خاکستری وارپ‌شده‌ی گرید (CLAHE)
    warped_original: تصویر خاکستری وارپ‌شده‌ی گرید (اصلی)
    output_dir: مسیر خروجی
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== ساخت سلول‌ها از هر دو تصویر =====
    cells = []
    for row in range(9):
        for col in range(9):
            # برش از تصویر CLAHE
            crop_clahe = make_raw_cell(warped, row, col)
            # برش از تصویر اصلی
            crop_original = make_raw_cell(warped_original, row, col)
            
            # اگر سلول خالی نبود، پردازش کن
            if not crop_clahe.is_empty:
                # پاس دادن هر دو تصویر به process_cell
                processed = process_cell(crop_clahe.image, crop_original.image)
                crop_clahe.image = processed
            
            cells.append(crop_clahe)

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

def process_cell(crop, crop_original=None, use_original=False) -> np.ndarray:
    """
    پردازش سلول با انتخاب خودکار تصویر مناسب بر اساس کنتراست
    
    پارامترها:
    - crop: تصویر سلول از warped (CLAHE شده)
    - crop_original: تصویر سلول از warped_original (عکس اصلی)
    - use_original: اگر True باشد، همیشه از crop_original استفاده می‌شود
    
    منطق:
    1. ابتدا کنتراست تصویر crop (CLAHE) بررسی می‌شود
    2. اگر کنتراست پایین بود → از crop_original با Adaptive Thresholding استفاده می‌شود
    3. اگر کنتراست خوب بود → از crop با OTSU استفاده می‌شود
    """
    
    # ===== مرحله 1: بررسی کنتراست تصویر CLAHE =====
    contrast = np.std(crop)
    
    # ===== مرحله 2: انتخاب تصویر و روش باینری‌سازی =====
    if contrast < 28 or use_original:
        # کنتراست پایین است → از تصویر اصلی استفاده کن
        if crop_original is None:
            # اگر تصویر اصلی وجود نداشت، از همان crop استفاده کن
            image_to_use = crop
        else:
            image_to_use = crop_original
        
        # فیلتر میانه
        denoised = cv2.medianBlur(image_to_use, 3)
        
        # Adaptive Thresholding
        _, binary = cv2.threshold(
            denoised,
            50,  # آستانه ثابت
            255,
            cv2.THRESH_BINARY
        )
    else:
        # کنتراست خوب است → از تصویر CLAHE استفاده کن
        # فیلتر میانه
        denoised = cv2.medianBlur(crop, 3)
        
        # OTSU Thresholding
        _, binary = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

    # ===== مرحله 3: اینورس شرطی =====
    white_ratio = cv2.countNonZero(binary) / binary.size
    if white_ratio > WHITE_RATIO_FOR_INVERSE:
        binary = cv2.bitwise_not(binary)

    h, w = binary.shape

    # ===== مرحله 4: حذف خطوط عمودی لبه =====
    remove_mask = np.zeros_like(binary)

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

            # فقط خطوط عمودی
            if dx <= 2:
                x = (x1 + x2) // 2
                if x <= margin_x or x >= w - margin_x:
                    cv2.line(remove_mask, (x1, y1), (x2, y2), 255, 3)

    binary = cv2.bitwise_and(binary, cv2.bitwise_not(remove_mask))

    # ===== مرحله 5: پاک‌سازی نویز =====
    binary = clean_noise(binary)
    
    # ===== مرحله 6: Closing =====
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary

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