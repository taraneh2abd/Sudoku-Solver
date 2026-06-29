from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import BOARD_SIZE
from src.utils import order_points, save_debug_image


@dataclass(frozen=True)
class BoardDetection:
    warped: np.ndarray
    corners: np.ndarray
    transform: np.ndarray


def preprocess_for_board(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    return threshold


def full_image_contour(image_shape: tuple[int, ...]) -> np.ndarray:
    height, width = image_shape[:2]
    return np.array(
        [
            [[0, 0]],
            [[width - 1, 0]],
            [[width - 1, height - 1]],
            [[0, height - 1]],
        ],
        dtype=np.int32,
    )


def find_board_contour(threshold: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = float(image_shape[0] * image_shape[1])

    for contour in contours[:40]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > image_area * 0.20:
            return approx

    foreground = cv2.findNonZero(closed)
    if foreground is not None:
        x, y, w, h = cv2.boundingRect(foreground)
        area_ratio = (w * h) / image_area
        aspect_ratio = w / float(h)
        if area_ratio > 0.55 and 0.75 <= aspect_ratio <= 1.25:
            return np.array(
                [
                    [[x, y]],
                    [[x + w - 1, y]],
                    [[x + w - 1, y + h - 1]],
                    [[x, y + h - 1]],
                ],
                dtype=np.int32,
            )

    return full_image_contour(image_shape)


def warp_board(image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = order_points(corners)
    destination = np.array(
        [
            [0, 0],
            [BOARD_SIZE - 1, 0],
            [BOARD_SIZE - 1, BOARD_SIZE - 1],
            [0, BOARD_SIZE - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(image, transform, (BOARD_SIZE, BOARD_SIZE))
    return warped, transform


def draw_projected_grid(image: np.ndarray, corners: np.ndarray, thickness: int = 4) -> np.ndarray:
    overlay = image.copy()
    source = np.array(
        [
            [0, 0],
            [BOARD_SIZE - 1, 0],
            [BOARD_SIZE - 1, BOARD_SIZE - 1],
            [0, BOARD_SIZE - 1],
        ],
        dtype="float32",
    )
    inverse_transform = cv2.getPerspectiveTransform(source, corners.astype("float32"))
    step = (BOARD_SIZE - 1) / 9.0

    for index in range(10):
        offset = index * step
        lines = np.array(
            [
                [[offset, 0], [offset, BOARD_SIZE - 1]],
                [[0, offset], [BOARD_SIZE - 1, offset]],
            ],
            dtype="float32",
        )
        projected = cv2.perspectiveTransform(lines, inverse_transform).astype(int)
        line_thickness = thickness + 2 if index % 3 == 0 else thickness
        cv2.line(overlay, tuple(projected[0, 0]), tuple(projected[0, 1]), (0, 0, 255), line_thickness)
        cv2.line(overlay, tuple(projected[1, 0]), tuple(projected[1, 1]), (0, 0, 255), line_thickness)

    return overlay


def detect_and_warp_board(image: np.ndarray, debug_dir: Path | None = None) -> BoardDetection:
    threshold = preprocess_for_board(image)
    contour = find_board_contour(threshold, image.shape)
    corners = order_points(contour)
    warped, transform = warp_board(image, corners)

    if debug_dir is not None:
        save_debug_image(debug_dir / "01_threshold.jpg", threshold)
        contour_debug = image.copy()
        cv2.drawContours(contour_debug, [contour], -1, (0, 255, 0), 3)
        save_debug_image(debug_dir / "02_board_contour.jpg", contour_debug)
        grid_debug = draw_projected_grid(image, corners)
        save_debug_image(debug_dir / "03_projected_grid_red.jpg", grid_debug)
        save_debug_image(debug_dir / "04_warped.jpg", warped)

    return BoardDetection(warped=warped, corners=corners, transform=transform)
