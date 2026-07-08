from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WARP_SIZE = 450
MIN_GRID_AREA_RATIO = 0.05


class GridNotFoundError(RuntimeError):
    """Kept for backwards compatibility with older callers. extract() no
    longer raises this in normal operation (see the 3-stage fallback in
    extract() below) but the class stays importable in case something else
    still catches it."""


def extract(pre, image, output_dir=None):
    """
    pre: a Preprocessed instance from src.preprocess.preprocess()
    image: the original (color or gray) image, only used here to draw the
           debug outline on top of it (same role `color` played in the
           original single-file extract_grid()).

    Three-stage detection, in priority order:
      1) PRIMARY  - our original method: median-blur + gaussian-blur +
         adaptive threshold on the (possibly downsampled) detect image,
         contour search with a Hough-line fallback. Unchanged from before.
      2) FALLBACK - only runs if (1) found nothing. Redoes the same
         contour+Hough search (_locate_grid) but on a fresh, full-resolution,
         NO-median-blur version of the image (matches the alternate
         "low accuracy" implementation).
      3) LAST RESORT - only if both (1) and (2) found nothing: treat the
         whole image as the grid instead of failing outright.
    """
    out_dir = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) PRIMARY (median blur + downsampling, as before) ----------
    corners, detector = _locate_grid(pre.detect_bin, pre.detect_blur)
    method = "primary"

    if corners is not None and pre.detect_scale < 1.0:
        corners = corners / pre.detect_scale

    # ---------- 2) FALLBACK (no median blur, full resolution) ----------
    if corners is None:
        print("[grid] primary method found nothing -> trying fallback (no median blur, full resolution)")
        fb_blurred = cv2.GaussianBlur(pre.normalized, (5, 5), 0)
        fb_binary = cv2.adaptiveThreshold(
            fb_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )
        if out_dir is not None:
            cv2.imwrite(str(out_dir / "05b_fallback_binary.png"), fb_binary)

        corners, detector = _locate_grid(fb_binary, fb_blurred)
        method = "fallback"

    # ---------- 3) LAST RESORT (whole image, never hard-fail) ----------
    if corners is None:
        print("[grid] fallback also found nothing -> using the full image as the grid")
        h, w = pre.gray.shape[:2]
        corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        method = "full_image_fallback"
        detector = "full_image"

    corners = _order_corners(corners)

    # the first set of corners we find is often not very accurate,
    # so we get the cornners of the first pass warp and then find the cornors again
    corners = _refine_corners(pre.normalized, corners)

    matrix = cv2.getPerspectiveTransform(corners, _warp_destination())
    inverse_matrix = np.linalg.inv(matrix)

    warped = cv2.warpPerspective(pre.normalized, matrix, (WARP_SIZE, WARP_SIZE))

    print(f"[grid] method used: {method} ({detector})")

    if out_dir is not None:
        if image.ndim == 3:
            color = image
        else:
            color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        outline = color.copy()
        cv2.polylines(outline, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        for point in corners.astype(int):
            cv2.circle(outline, tuple(point), 8, (0, 0, 255), -1)
        cv2.putText(
            outline, f"method: {method}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
        )
        cv2.imwrite(str(out_dir / "05_grid_outline.png"), outline)
        cv2.imwrite(str(out_dir / "06_warped.png"), warped)

    return corners, warped, inverse_matrix


def _warp_destination() -> np.ndarray:
    # get the corners of the warped grid in the order (tl, tr, br, bl)
    return np.array(
        [[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]],
        dtype=np.float32,
    )


def _refine_corners(normalized, corners) -> np.ndarray:
    matrix = cv2.getPerspectiveTransform(corners, _warp_destination())
    warped = cv2.warpPerspective(normalized, matrix, (WARP_SIZE, WARP_SIZE))

    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(warped, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10,
    )

    # isolate long horizontal and vertical lines (about 45 pixels long)
    length = WARP_SIZE // 10
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)

    # Combine them.
    grid_lines = cv2.bitwise_or(horizontal, vertical)
    grid_lines = cv2.dilate(grid_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(grid_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return corners

    largest_contour = max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])
    x, y, w, h = cv2.boundingRect(largest_contour)

    if w < 0.5 * WARP_SIZE or h < 0.5 * WARP_SIZE:
        return corners

    if w > 0.95 * WARP_SIZE and h > 0.95 * WARP_SIZE and x < 0.05 * WARP_SIZE and y < 0.05 * WARP_SIZE:
        return corners

    perimeter = cv2.arcLength(largest_contour, True)

    for epsilon in (0.02, 0.05, 0.1):
        quad = cv2.approxPolyDP(largest_contour, epsilon * perimeter, True)
        if len(quad) == 4 and cv2.isContourConvex(quad):
            refined = cv2.perspectiveTransform(
                quad.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(matrix)
            ).reshape(4, 2)
            return _order_corners(refined)

    quad = np.array([
        [x, y], [x + w, y], [x + w, y + h], [x, y + h]
    ], dtype=np.float32)

    refined = cv2.perspectiveTransform(
        quad.reshape(-1, 1, 2), np.linalg.inv(matrix)
    ).reshape(4, 2)

    return _order_corners(refined)


def _locate_grid(binary, blurred) -> np.ndarray | None:
    min_area = MIN_GRID_AREA_RATIO * binary.size

    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.02, 0.05, 0.1):
            quad = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(quad) == 4 and cv2.isContourConvex(quad):
                return quad.reshape(4, 2).astype(np.float32), "contour"

    corners = _corners_from_hough(blurred, min_area)
    if corners is not None:
        return corners, "hough"

    return None, None

def _corners_from_hough(blurred, min_area) -> np.ndarray | None:
    print("herehereedfjkahfjhfhjdhfkakj")

    edges = cv2.Canny(blurred, 50, 150)
    threshold = max(80, min(blurred.shape) // 4)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    if lines is None:
        return None

    horizontal = []
    vertical = []

    for dist_to_tl, theta in lines[:, 0]:
        if dist_to_tl < 0:
            dist_to_tl, theta = -dist_to_tl, theta - np.pi

        if abs(theta) < np.pi / 4:
            vertical.append((dist_to_tl, theta))

        elif abs(theta - np.pi / 2) < np.pi / 4:
            horizontal.append((dist_to_tl, theta))

    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    top, bottom = min(horizontal), max(horizontal)
    left, right = min(vertical), max(vertical)
    points = []

    for pair in ((top, left), (top, right), (bottom, right), (bottom, left)):
        point = _intersect_lines(*pair)
        if point is None:
            return None
        points.append(point)
    corners = np.array(points, dtype=np.float32)

    if cv2.contourArea(corners) < min_area:
        return None
    if not cv2.isContourConvex(corners.astype(np.int32)):
        return None
    return corners


def _intersect_lines(line_a, line_b) -> tuple[float, float] | None:
    (rho_a, theta_a), (rho_b, theta_b) = line_a, line_b
    coefficients = np.array(
        [[np.cos(theta_a), np.sin(theta_a)], [np.cos(theta_b), np.sin(theta_b)]]
    )
    if abs(np.linalg.det(coefficients)) < 1e-8:
        return None
    x, y = np.linalg.solve(coefficients, np.array([rho_a, rho_b]))
    return float(x), float(y)


def _order_corners(points) -> np.ndarray:
    # 4 points as (tl, tr, br, bl) — invariant to rotations below 45 degrees
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    return np.array(
        [
            points[sums.argmin()],   # top-left: smallest x + y
            points[diffs.argmin()],  # top-right: smallest y - x
            points[sums.argmax()],   # bottom-right: largest x + y
            points[diffs.argmax()],  # bottom-left: largest y - x
        ],
        dtype=np.float32,
    )
