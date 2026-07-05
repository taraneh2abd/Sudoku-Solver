from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_DETECT_DIM = 1024  # downsample target for grid detection (block size 11 is good for 500–1024px images)


@dataclass
class Preprocessed:
    gray: np.ndarray
    normalized: np.ndarray
    blurred: np.ndarray
    binary: np.ndarray
    detect_blur: np.ndarray
    detect_bin: np.ndarray
    detect_scale: float


def preprocess(image, output_dir=None) -> Preprocessed:
    # if not gray, make gray
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # some robustness stuff
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    denoised = cv2.medianBlur(normalized, 5)
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )

    detect_scale = min(1.0, MAX_DETECT_DIM / max(gray.shape[:2]))
    if detect_scale < 1.0:
        dh = int(gray.shape[0] * detect_scale)
        dw = int(gray.shape[1] * detect_scale)
        detect_gray = cv2.resize(gray, (dw, dh))
        detect_norm = clahe.apply(detect_gray)
        detect_blur = cv2.GaussianBlur(cv2.medianBlur(detect_norm, 5), (5, 5), 0)
        detect_bin = cv2.adaptiveThreshold(
            detect_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )
    else:
        detect_blur = blurred
        detect_bin = binary

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "01_gray.png"), gray)
        cv2.imwrite(str(out_dir / "02_clahe.png"), normalized)
        cv2.imwrite(str(out_dir / "03_denoised.png"), blurred)
        cv2.imwrite(str(out_dir / "04_binary.png"), binary)

    return Preprocessed(
        gray=gray,
        normalized=normalized,
        blurred=blurred,
        binary=binary,
        detect_blur=detect_blur,
        detect_bin=detect_bin,
        detect_scale=detect_scale,
    )
