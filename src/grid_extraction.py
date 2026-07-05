"""
Grid extraction module
-----------------------
Two methods:
1. LOW accuracy : EXACTLY the second code's method (contour-based with Hough fallback and corner refinement)
2. HIGH accuracy: Canny -> LSD (gradient filtered) -> segment intersections ->
   custom Hough (gradient-direction gated) -> filter Hough lines
   by intersection count -> pick top/bottom/left/right -> corners.

Both paths warp the found quad to a square and draw a 9x9 grid (10 lines
each direction -> 81 cells) on the warped image.

When accuracy == "high", both results are computed and the one whose grid
lines overlap more with the LSD segments (in the ORIGINAL image space) is
kept as the final result.
"""

from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import maximum_filter

# ============================================================================
# CONSTANTS
# ============================================================================

WARP_SIZE = 450  # از کد دوم
GRID_N = 10      # 10 lines each direction -> 9x9 = 81 cells
OVERLAP_TOL_PX = 5

# Parameters for LOW method (exactly from second code)
MIN_GRID_AREA_RATIO = 0.05
HOUGH_THRESHOLD_MULTIPLIER = 4

# Parameters for HIGH method
GRAD_THRESHOLD = 0.62
DIR_THRESHOLD = 0.4
HOUGH_VOTES = 60
INTER_TOL_PX = 10
MIN_INTER_ON_LINE = 7


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save(path: Path, img) -> None:
    cv2.imwrite(str(path), img)
    print(f"  saved -> {path.name}")


def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype="float32")


def warp(image, corners):
    ordered = order_points(corners)
    dst = np.array([
        [0, 0], [WARP_SIZE - 1, 0],
        [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    M_inv = np.linalg.inv(M)
    warped = cv2.warpPerspective(image, M, (WARP_SIZE, WARP_SIZE))
    return warped, M_inv


def grid_lines_warped():
    """Return 9x9 grid lines (10 each direction) in warped-image coordinates."""
    lines = []
    step = WARP_SIZE / (GRID_N - 1)
    for i in range(GRID_N):
        x = int(round(i * step))
        lines.append(((x, 0), (x, WARP_SIZE - 1)))          # vertical
        y = int(round(i * step))
        lines.append(((0, y), (WARP_SIZE - 1, y)))           # horizontal
    return lines


def draw_grid(warped_img, lines):
    dbg = warped_img.copy()
    if dbg.ndim == 2:
        dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
    for p1, p2 in lines:
        cv2.line(dbg, p1, p2, (0, 0, 255), 1)
    return dbg


def draw_intersections(img, points, color=(0, 255, 255)):
    """Draw intersection points on image."""
    dbg = img.copy()
    if dbg.ndim == 2:
        dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
    for pt in points:
        cv2.circle(dbg, (int(pt[0]), int(pt[1])), 3, color, -1)
    return dbg


# ============================================================================
# METHOD 1: LOW ACCURACY (EXACTLY THE SECOND CODE)
# ============================================================================

def _intersect_lines(line_a, line_b) -> tuple[float, float] | None:
    """Find intersection of two lines in Hesse normal form."""
    (rho_a, theta_a), (rho_b, theta_b) = line_a, line_b
    coefficients = np.array(
        [[np.cos(theta_a), np.sin(theta_a)], [np.cos(theta_b), np.sin(theta_b)]]
    )
    if abs(np.linalg.det(coefficients)) < 1e-8:
        return None
    x, y = np.linalg.solve(coefficients, np.array([rho_a, rho_b]))
    return float(x), float(y)


def _corners_from_hough(blurred, min_area) -> np.ndarray | None:
    """Simple Hough transform fallback from the second code."""
    edges = cv2.Canny(blurred, 50, 150)
    threshold = max(80, min(blurred.shape) // HOUGH_THRESHOLD_MULTIPLIER)
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


def _refine_corners(normalized, corners) -> np.ndarray:
    """Corner refinement from the second code."""
    dst = np.array([
        [0, 0], [WARP_SIZE - 1, 0],
        [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1],
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(normalized, matrix, (WARP_SIZE, WARP_SIZE))

    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(warped, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10,
    )

    # isolate long horizontal and vertical lines
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
            return order_points(refined)
            
    quad = np.array([
        [x, y], [x + w, y], [x + w, y + h], [x, y + h]
    ], dtype=np.float32)
    
    refined = cv2.perspectiveTransform(
        quad.reshape(-1, 1, 2), np.linalg.inv(matrix)
    ).reshape(4, 2)
    
    return order_points(refined)


def extract_low_accuracy(pre_img, color_img, output_dir):
    """
    LOW ACCURACY: EXACTLY the second code's method.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[Low] Finding contours (exactly like second code)")
    
    # Step 1: Prepare image like second code
    min_area = MIN_GRID_AREA_RATIO * pre_img.size
    
    # Use CLAHE and blur like second code
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(pre_img)
    # denoised = cv2.medianBlur(normalized, 5)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )
    save(out / "low_step1_binary.jpg", binary)

    # Step 2: Find contours (like second code)
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    save(out / "low_step2_closed.jpg", closed)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    corners = None
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.02, 0.05, 0.1):
            quad = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(quad) == 4 and cv2.isContourConvex(quad):
                corners = quad.reshape(4, 2).astype(np.float32)
                break
        if corners is not None:
            break

    # Step 3: Fallback to Hough if contour failed (like second code)
    if corners is None:
        print("  [Low] Contour failed, trying Hough transform fallback")
        corners = _corners_from_hough(blurred, min_area)
        
        if corners is not None:
            # Draw Hough lines for debugging
            edges = cv2.Canny(blurred, 50, 150)
            threshold = max(80, min(blurred.shape) // HOUGH_THRESHOLD_MULTIPLIER)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
            if lines is not None:
                hough_dbg = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2BGR)
                for rho, theta in lines[:, 0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(hough_dbg, (x1, y1), (x2, y2), (0, 255, 0), 1)
                save(out / "low_hough_lines.jpg", hough_dbg)

    # Step 4: Draw initial corners
    dbg = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        pts = order_points(corners).astype(int)
        for i in range(4):
            cv2.line(dbg, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 0, 255), 3)
    save(out / "low_step3_initial_corners.jpg", dbg)

    # Step 5: Refine corners (like second code)
    if corners is None:
        print("  [Low] no quad found -> using full image as fallback")
        h, w = pre_img.shape[:2]
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    else:
        print("  [Low] Refining corners")
        corners = _refine_corners(normalized, corners)

    # Step 6: Draw refined corners
    dbg2 = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2BGR)
    pts = order_points(corners).astype(int)
    for i in range(4):
        cv2.line(dbg2, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 3)
    save(out / "low_step4_refined_corners.jpg", dbg2)

    # Step 7: Warp and draw grid
    warped, M_inv = warp(color_img, corners)
    save(out / "low_step5_warped.jpg", warped)

    lines = grid_lines_warped()
    dbg_grid = draw_grid(warped, lines)
    save(out / "low_step6_grid.jpg", dbg_grid)

    return corners, warped, (lines, M_inv)


# ============================================================================
# METHOD 2: HIGH ACCURACY (LSD + Custom Hough)
# ============================================================================

def _seg_gradient_score(seg, gx, gy, mag):
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    nx, ny = -dy / length, dx / length
    n_samples = max(5, int(length / 4))
    ts = np.linspace(0, 1, n_samples)
    xs = np.clip((x1 + ts * dx).astype(int), 0, gx.shape[1] - 1)
    ys = np.clip((y1 + ts * dy).astype(int), 0, gx.shape[0] - 1)
    m = mag[ys, xs]
    strong = m > 10.0
    if not np.any(strong):
        return 0.0
    ugx = gx[ys[strong], xs[strong]] / (m[strong] + 1e-8)
    ugy = gy[ys[strong], xs[strong]] / (m[strong] + 1e-8)
    alignment = np.abs(ugx * nx + ugy * ny)
    return float(np.mean(alignment))


def _compute_gradient_direction(gray_img):
    gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    return np.cos(np.arctan2(gy, gx))


def _segments_intersect(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-8:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    eps = 0.05
    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        return p1 + t * d1
    return None


def _rho_theta_to_endpoints(rho, theta, w, h, extension=3000):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    if abs(sin_t) > 1e-6:
        x0, y0 = 0, int((rho - 0 * cos_t) / sin_t)
        x1, y1 = w - 1, int((rho - (w - 1) * cos_t) / sin_t)
    else:
        x0, y0 = int(rho / (cos_t + 1e-12)), 0
        x1, y1 = int(rho / (cos_t + 1e-12)), h - 1
    dx, dy = x1 - x0, y1 - y0
    length = max(np.hypot(dx, dy), 1e-6)
    ex = int(extension * dx / length)
    ey = int(extension * dy / length)
    return (x0 - ex, y0 - ey), (x1 + ex, y1 + ey)


def _rho_theta_intersection(rho1, theta1, rho2, theta2):
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-10:
        return None
    x = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
    y = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det
    return (float(x), float(y))


def _representative_point(rho, theta, cx, cy):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    d = cx * cos_t + cy * sin_t - rho
    return float(cx - d * cos_t), float(cy - d * sin_t)


def extract_high_accuracy(pre_img, color_img, output_dir):
    """
    HIGH ACCURACY: LSD + Custom Hough with gradient gating.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[High] Canny + Sobel")
    edges = cv2.Canny(pre_img, 50, 150, apertureSize=3)
    save(out / "high_step1_canny.jpg", edges)
    
    gx = cv2.Sobel(pre_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(pre_img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    print("[High] LSD + gradient filter")
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines_lsd, _, _, _ = lsd.detect(pre_img)
    all_segs = lines_lsd.reshape(-1, 4).tolist() if lines_lsd is not None else []
    kept_segs = [seg for seg in all_segs
                 if _seg_gradient_score(seg, gx, gy, mag) > GRAD_THRESHOLD]
    
    dbg_lsd = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2BGR)
    for seg in kept_segs:
        x1, y1, x2, y2 = [int(v) for v in seg]
        cv2.line(dbg_lsd, (x1, y1), (x2, y2), (0, 255, 0), 2)
    save(out / "high_step2_lsd_filtered.jpg", dbg_lsd)

    print("[High] Segment intersections")
    segs = [np.array(s, dtype="float64") for s in kept_segs]
    inter_pts = []
    for i in range(len(segs)):
        p1, p2 = segs[i][:2], segs[i][2:]
        for j in range(i + 1, len(segs)):
            p3, p4 = segs[j][:2], segs[j][2:]
            pt = _segments_intersect(p1, p2, p3, p4)
            if pt is not None:
                inter_pts.append(pt)
    
    # Draw intersections
    dbg_inter = draw_intersections(pre_img, inter_pts, (0, 255, 255))
    save(out / "high_step3_intersections.jpg", dbg_inter)

    print("[High] Custom Hough (gradient-direction gated)")
    h, w = edges.shape
    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
    n_rho = 2 * diag + 1
    num_thetas = 180
    thetas = np.deg2rad(np.linspace(0, 180, num_thetas, endpoint=False))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    direction = _compute_gradient_direction(pre_img)
    accum = np.zeros((n_rho, num_thetas), dtype=np.int32)
    edge_yx = np.argwhere(edges > 0)
    for (y, x) in edge_yx:
        dir_val = direction[y, x]
        for t_idx in range(num_thetas):
            cos_t = cos_thetas[t_idx]
            if abs(cos_t - dir_val) < DIR_THRESHOLD:
                rho = x * cos_t + y * sin_thetas[t_idx]
                rho_idx = int(round(rho)) + diag
                if 0 <= rho_idx < n_rho:
                    accum[rho_idx, t_idx] += 1

    loc_max = (accum == maximum_filter(accum, size=20))
    peaks = np.argwhere(loc_max & (accum >= HOUGH_VOTES))
    hough_lines = []
    for (rho_idx, t_idx) in peaks:
        hough_lines.append((float(rho_idx - diag), float(thetas[t_idx]), int(accum[rho_idx, t_idx])))
    hough_lines.sort(key=lambda x: x[2], reverse=True)
    print(f"  Hough candidate lines: {len(hough_lines)}")

    # Draw all Hough lines
    dbg_hough = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2BGR)
    for rho, theta, votes in hough_lines[:20]:  # Show top 20
        p1, p2 = _rho_theta_to_endpoints(rho, theta, w, h)
        cv2.line(dbg_hough, p1, p2, (0, 255, 0), 1)
    save(out / "high_step4_hough_lines.jpg", dbg_hough)

    print("[High] Filter Hough lines by intersection count")
    counts = [0] * len(hough_lines)
    for li, (rho_l, theta_l, _) in enumerate(hough_lines):
        cos_t, sin_t = np.cos(theta_l), np.sin(theta_l)
        for pt in inter_pts:
            if abs(pt[0] * cos_t + pt[1] * sin_t - rho_l) <= INTER_TOL_PX:
                counts[li] += 1
    filtered_lines = [hough_lines[i] for i in range(len(hough_lines)) if counts[i] >= MIN_INTER_ON_LINE]
    print(f"  filtered Hough lines: {len(filtered_lines)}")

    # Draw filtered Hough lines
    dbg_filtered = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2BGR)
    for rho, theta, votes in filtered_lines:
        p1, p2 = _rho_theta_to_endpoints(rho, theta, w, h)
        cv2.line(dbg_filtered, p1, p2, (0, 255, 255), 2)
    save(out / "high_step5_filtered_lines.jpg", dbg_filtered)

    pool = filtered_lines if len(filtered_lines) >= 4 else hough_lines[:4]
    if len(pool) < 4:
        print("  [High] not enough lines found")
        return None, None, None

    cx, cy = w / 2.0, h / 2.0
    top = min(pool, key=lambda l: _representative_point(l[0], l[1], cx, cy)[1])
    bottom = max(pool, key=lambda l: _representative_point(l[0], l[1], cx, cy)[1])
    left = min(pool, key=lambda l: _representative_point(l[0], l[1], cx, cy)[0])
    right = max(pool, key=lambda l: _representative_point(l[0], l[1], cx, cy)[0])

    # Draw the 4 selected lines
    dbg_four = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for idx, line in enumerate((top, bottom, left, right)):
        p1, p2 = _rho_theta_to_endpoints(line[0], line[1], w, h)
        cv2.line(dbg_four, p1, p2, colors[idx], 3)
        # Add label
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        labels = ["TOP", "BOTTOM", "LEFT", "RIGHT"]
        cv2.putText(dbg_four, labels[idx], mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2)
    save(out / "high_step6_four_lines.jpg", dbg_four)

    print("[High] Computing corners from line intersections")
    tl = _rho_theta_intersection(top[0], top[1], left[0], left[1])
    tr = _rho_theta_intersection(top[0], top[1], right[0], right[1])
    br = _rho_theta_intersection(bottom[0], bottom[1], right[0], right[1])
    bl = _rho_theta_intersection(bottom[0], bottom[1], left[0], left[1])
    
    if any(p is None for p in (tl, tr, br, bl)):
        print("  [High] could not compute corners")
        return None, None, None

    corners = np.array([list(tl), list(tr), list(br), list(bl)], dtype="float32")

    # Draw corners
    dbg_corners = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2BGR)
    pts = order_points(corners).astype(int)
    for i in range(4):
        cv2.line(dbg_corners, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 0, 255), 3)
        cv2.circle(dbg_corners, tuple(pts[i]), 8, (0, 255, 255), -1)
    # Add corner labels
    labels = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(pts):
        cv2.putText(dbg_corners, labels[i], (pt[0] + 10, pt[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    save(out / "high_step7_corners.jpg", dbg_corners)

    warped, M_inv = warp(color_img, corners)
    save(out / "high_step8_warped.jpg", warped)

    lines = grid_lines_warped()
    dbg_grid = draw_grid(warped, lines)
    save(out / "high_step9_grid.jpg", dbg_grid)

    return corners, warped, (lines, M_inv, kept_segs)


# ============================================================================
# COMPARISON
# ============================================================================

def _point_to_segments_min_dist(pt, segs):
    px, py = pt
    best = float("inf")
    for seg in segs:
        x1, y1, x2, y2 = seg
        dx, dy = x2 - x1, y2 - y1
        length2 = dx * dx + dy * dy
        if length2 < 1e-9:
            d = np.hypot(px - x1, py - y1)
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length2))
            cx, cy = x1 + t * dx, y1 + t * dy
            d = np.hypot(px - cx, py - cy)
        if d < best:
            best = d
    return best


def _overlap_score(lines_warped, M_inv, lsd_segs):
    """Map sampled points of warped grid lines back to original image space
    and count how many lie within OVERLAP_TOL_PX of an LSD segment."""
    score = 0
    for p1, p2 in lines_warped:
        n_samples = 20
        for t in np.linspace(0, 1, n_samples):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            src = M_inv @ np.array([x, y, 1.0])
            src = src[:2] / src[2]
            if _point_to_segments_min_dist(src, lsd_segs) <= OVERLAP_TOL_PX:
                score += 1
    return score


# ============================================================================
# MAIN EXTRACT FUNCTION
# ============================================================================

def extract(pre_img, color_img, output_dir, accuracy="high"):
    """
    Main entry point.
    
    Args:
        pre_img: Preprocessed grayscale image
        color_img: Original color image for warping
        output_dir: Directory to save outputs
        accuracy: 'low' or 'high'
            - 'low': Exactly the second code's method
            - 'high': Both methods, choose best by overlap score
    
    Returns:
        corners, warped, (lines, M_inv) or (lines, M_inv, kept_segs)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Always compute LOW method (exactly second code)
    print("\n" + "="*60)
    print("LOW ACCURACY METHOD (exactly second code)")
    print("="*60)
    low_result = extract_low_accuracy(pre_img, color_img, output_dir)
    corners_low, warped_low, (lines_low, M_inv_low) = low_result

    if accuracy == "low":
        return low_result

    # Compute HIGH method
    print("\n" + "="*60)
    print("HIGH ACCURACY METHOD (LSD + Custom Hough)")
    print("="*60)
    high_result = extract_high_accuracy(pre_img, color_img, output_dir)

    # If high failed, fallback to low
    if high_result[0] is None:
        print("\n[Compare] High accuracy failed, falling back to low accuracy")
        return low_result

    corners_high, warped_high, (lines_high, M_inv_high, lsd_segs) = high_result

    # Compare scores
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    score_low = _overlap_score(lines_low, M_inv_low, lsd_segs)
    score_high = _overlap_score(lines_high, M_inv_high, lsd_segs)
    
    print(f"  LOW method overlap score:  {score_low}")
    print(f"  HIGH method overlap score: {score_high}")
    
    # Determine best
    if score_high >= score_low:
        print(f"\n[Compare] HIGH method wins (score={score_high} >= {score_low})")
        best_result = high_result
        best_name = "HIGH"
    else:
        print(f"\n[Compare] LOW method wins (score={score_low} > {score_high})")
        best_result = low_result
        best_name = "LOW"

    # Save comparison visualization
    compare_img = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2BGR)
    
    # Draw LOW in red
    if corners_low is not None:
        pts = order_points(corners_low).astype(int)
        for i in range(4):
            cv2.line(compare_img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 0, 255), 2)
        center = np.mean(pts, axis=0).astype(int)
        cv2.putText(compare_img, "LOW", (center[0]-20, center[1]+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw HIGH in green
    if corners_high is not None:
        pts = order_points(corners_high).astype(int)
        for i in range(4):
            cv2.line(compare_img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
        center = np.mean(pts, axis=0).astype(int)
        cv2.putText(compare_img, "HIGH", (center[0]-20, center[1]+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Mark winner
    cv2.putText(compare_img, f"WINNER: {best_name}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    save(out / "comparison_both_methods.jpg", compare_img)

    return best_result