from __future__ import annotations

import cv2
import numpy as np

from src.config import BOARD_SIZE, CELL_SIZE


def overlay_solution(
    original: np.ndarray,
    board_corners: np.ndarray,
    initial_grid: np.ndarray,
    solved_grid: np.ndarray,
) -> np.ndarray:
    answer_layer = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)

    for row in range(9):
        for col in range(9):
            if int(initial_grid[row, col]) != 0:
                continue
            value = str(int(solved_grid[row, col]))
            x = col * CELL_SIZE + CELL_SIZE // 3
            y = row * CELL_SIZE + int(CELL_SIZE * 0.72)
            cv2.putText(answer_layer, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 90, 255), 2, cv2.LINE_AA)

    source = np.array(
        [
            [0, 0],
            [BOARD_SIZE - 1, 0],
            [BOARD_SIZE - 1, BOARD_SIZE - 1],
            [0, BOARD_SIZE - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(source, board_corners.astype("float32"))
    projected = cv2.warpPerspective(answer_layer, transform, (original.shape[1], original.shape[0]))
    mask = cv2.cvtColor(projected, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

    output = original.copy()
    output[mask > 0] = projected[mask > 0]
    return output
