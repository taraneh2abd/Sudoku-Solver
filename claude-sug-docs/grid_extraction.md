# `src/grid_extraction.py` — Reference

## Fix log (2026-07-08)

All robustness findings below were addressed directly in `src/grid_extraction.py`:

- Removed the leftover debug `print("herehereedfjkahfjhfhjdhfkakj")` in `_corners_from_hough`.
- Replaced the fixed `cv2.Canny(blurred, 50, 150)` thresholds with a new `_auto_canny()`
  helper that derives lower/upper thresholds from the image's own median intensity, so
  contrast/lighting differences (shadows, low-contrast scans) don't need hand-tuned constants.
- The FALLBACK stage no longer drops median blur entirely (which was actively
  counter-productive for salt-and-pepper noise, the exact noise type median blur fights). It
  now uses a lighter `cv2.medianBlur(pre.normalized, 3)` before the Gaussian blur — still
  meaningfully different from PRIMARY's `medianBlur(..., 5)`, but no longer removes the one
  filter that helps with `_sp` test cases.
- Rewrote `_order_corners()` to sort the 4 points by angle around their own centroid instead
  of by `x+y`/`y-x` extremes. The old method's "invariant below 45°" limit was real: verified
  with a synthetic sweep (elongated quad, rotated 0–180° at an off-origin position) that it
  produced 8 self-intersecting ("bowtie") orderings out of 148 rotation/permutation trials,
  vs. 0 for the new angle-sort method. See the commit/PR for the verification script. Note
  this still cannot recover which corner was the *printed* top-left after an exact 90/180/270°
  rotation (that needs content-based cues, not just geometry) — it only guarantees a valid,
  non-self-intersecting corner order for the perspective warp.

Not changed: `MIN_GRID_AREA_RATIO`, the Hough vote threshold formula, and the two-bucket
(horizontal/vertical) line classification in `_corners_from_hough` are all still fixed/simple;
they were lower-priority findings and are left as future-work candidates below.

## Purpose

Takes the `Preprocessed` bundle from `src/preprocess.py` plus the original
image, locates the sudoku grid's 4 outer corners, and perspective-warps it to
a fixed `WARP_SIZE x WARP_SIZE` square for `src/cell_extraction.py`. Returns
`(corners, warped, inverse_matrix)`. `extract()` is the only public entry
point; everything else in the file is a private helper (`_`-prefixed).

## 3-stage detection strategy (`extract()`, lines 19-103)

1. **PRIMARY** (lines 42-47): calls `_locate_grid(pre.detect_bin,
   pre.detect_blur)` — i.e. the (possibly downsampled) median-blur +
   Gaussian-blur + adaptive-threshold image from `preprocess.py`. If a corner
   set is found and the image was downsampled (`pre.detect_scale < 1.0`),
   corners are divided by `detect_scale` to map back to full-resolution
   coordinates.
2. **FALLBACK** (lines 49-61): only runs if PRIMARY returned `None`. Rebuilds
   a *fresh* blur+threshold pair from `pre.normalized` (full resolution,
   **no median blur** — `fb_blurred = cv2.GaussianBlur(pre.normalized, (5,5),
   0)` then `adaptiveThreshold(fb_blurred, ..., 11, 2)`), then re-runs the same
   `_locate_grid`.
3. **LAST RESORT / full-image fallback** (lines 63-71): only if both above
   returned `None`. Never raises — treats the whole image's 4 corners
   `[[0,0],[w-1,0],[w-1,h-1],[0,h-1]]` as the grid instead of failing. Confirms
   the docstring's claim that `extract()` "no longer raises `GridNotFoundError`
   in normal operation."

After whichever stage succeeds, corners are always run through
`_order_corners()` then `_refine_corners()` (lines 73-77) before the final
`cv2.getPerspectiveTransform` + `cv2.warpPerspective` to `WARP_SIZE x
WARP_SIZE`.

### Per-function role

- **`_locate_grid(binary, blurred)`** (lines 170-192): the core single-pass
  detector used identically by both PRIMARY and FALLBACK (they just feed it
  different images). First tries **contour search**: `MORPH_CLOSE` (3x3) on
  `binary`, take the 5 largest external contours above `MIN_GRID_AREA_RATIO *
  binary.size`, try `approxPolyDP` at epsilons `(0.02, 0.05, 0.1)` looking for
  a convex quadrilateral. If no contour yields a quad, falls back to
  **`_corners_from_hough`** on `blurred`.
- **`_corners_from_hough(blurred, min_area)`** (lines 194-235): Canny edges
  (fixed 50/150 thresholds) → `cv2.HoughLines` (fixed vote threshold) →
  buckets each line into `horizontal`/`vertical` by its `theta` → picks
  extreme lines (`min`/`max` by rho) as the 4 grid sides → intersects them
  pairwise into 4 corner points → validates area/convexity.
- **`_intersect_lines(line_a, line_b)`** (lines 238-246): solves the 2x2
  linear system from each line's `(rho, theta)` normal form
  (`x*cos(theta) + y*sin(theta) = rho`) for the intersection point; returns
  `None` if the two lines are near-parallel (`det < 1e-8`).
- **`_order_corners(points)`** (lines 249-261): reorders an arbitrary 4-point
  array into `(tl, tr, br, bl)` using the classic sum/diff trick: min `x+y` =
  top-left, max `x+y` = bottom-right, min `y-x` = top-right, max `y-x` =
  bottom-left.
- **`_refine_corners(normalized, corners)`** (lines 114-167): does a *first*
  perspective warp using the roughly-located corners, then re-detects the grid
  border on the warped square using morphological horizontal/vertical line
  extraction (`MORPH_OPEN` with 45px-long rectangular kernels), finds the
  largest bounding contour, and re-derives (via inverse-warp) a tighter set of
  original-image corners. Falls back to the input `corners` unchanged if
  nothing better is found, if the found box is too small (`< 0.5 *
  WARP_SIZE`), or if it's basically the full warped image already (`> 0.95 *
  WARP_SIZE` and near origin — meaning "nothing to refine").

## Key constants

- `WARP_SIZE = 450` — output resolution of the warped grid; also the value
  `_refine_corners` uses to derive its "long line" kernel length (`WARP_SIZE //
  10 = 45px`) and its sanity-check thresholds (`0.5 *`, `0.95 *`, `0.05 *
  WARP_SIZE`).
- `MIN_GRID_AREA_RATIO = 0.05` — a contour (or Hough-derived quad) must cover
  at least 5% of the *detection* image's area to be accepted as the grid. This
  is a single fixed fraction regardless of how far away/zoomed-out the photo
  was taken.

## Known Issues / Robustness Gaps

### 1. Fixed Canny thresholds `(50, 150)` in `_corners_from_hough` (line 197)
- **Triggers on:** low-contrast regions from `add_shadow` (`data/bad-sodu/*_shadow*`),
  which multiplies pixel intensity by a linspace factor as low as `0.25` on one
  side of the image (see `tests/create-noisy-tests.py`'s `add_shadow`).
- **Why it fails:** Canny's gradient magnitude on the shadowed side shrinks
  roughly in proportion to the local intensity scale, so real grid-line edges
  there can fall below the fixed lower threshold of 50 while edges on the
  bright side are found normally — producing a broken/one-sided edge map where
  Hough sees only 2-3 of the 4 grid sides clearly.
- **Also triggers on:** heavy salt-and-pepper noise (`add_salt_pepper` flips 8%
  of pixels to pure 0/255, see `amount = 0.08`). Fixed low/high thresholds
  don't adapt to the resulting massive spurious-gradient count, so Canny
  produces a dense fog of tiny edges alongside the real grid edges.
- **Suggested fix:** replace fixed thresholds with median-based auto-Canny
  (`lower = 0.66*median, upper = 1.33*median` on the blurred image, a
  well-known OpenCV idiom), or estimate per-image contrast (e.g. std-dev of
  `blurred`) and scale the two thresholds accordingly.

### 2. Fixed Hough vote threshold `max(80, min(blurred.shape)//4)` (line 198)
- **Triggers on:** salt-and-pepper or Gaussian noise (`std=40` in
  `add_gaussian_noise`) fragmenting what would be one long straight edge into
  many short collinear segments after Canny.
- **Why it fails:** `cv2.HoughLines`'s accumulator counts edge pixels
  consistent with a given `(rho, theta)` bin; fragmenting the true grid line
  doesn't necessarily reduce the total vote count much (broken segments still
  vote for the same bin), but noise pixels scattered near-collinear by chance
  can also accumulate votes, and real vote counts along partially-shadowed or
  partially-occluded lines can drop under the fixed threshold. There's no
  adjustment for the fact that a rotated grid's line may have a shorter
  in-frame projection than an axis-aligned one — the required vote count
  doesn't scale with the line's actual pixel length, only with overall image
  size.
- **Suggested fix:** use `cv2.HoughLinesP` (probabilistic) with
  `minLineLength`/`maxLineGap` tuned to expected grid-cell size, or make the
  threshold proportional to the *detected contour's* perimeter/area from the
  earlier contour pass rather than a flat fraction of image dimension, or try
  multiple thresholds in a descending loop the way `approxPolyDP` already
  tries multiple epsilons.

### 3. Fixed adaptive-threshold `blockSize=11, C=2` (preprocess.py + fallback at grid_extraction.py:55)
- **Triggers on:** shadow images where illumination varies smoothly but the
  variation is large-scale (spans the whole image width/height), and on
  salt-and-pepper images where an 11x11 neighborhood can contain enough
  flipped pixels to shift the local mean and mis-threshold real grid pixels.
- **Why it fails:** a single fixed block size can't simultaneously suppress
  large-scale shadow gradients and stay small enough to preserve thin grid
  lines under dense impulsive noise — 11px was tuned (per the comment in
  `preprocess.py`) for images in the 500-1024px detect range, not for any
  particular noise regime.
- **Suggested fix:** pre-denoise more aggressively (e.g. `cv2.medianBlur` with
  a size chosen from measured salt-and-pepper density, or `cv2.fastNlMeansDenoising`)
  before thresholding when noise is high, and/or normalize illumination first
  (e.g. divide by a heavily-blurred version of the image, a common "flat-field"
  shadow-removal trick) before adaptive threshold rather than relying on CLAHE
  + a small fixed block size alone.

### 4. `_order_corners`'s "invariant to rotations below 45 degrees" claim (line 250) is a boundary condition, not a real robustness guarantee
- **What it does:** classic sum/diff corner sort — `argmin(x+y)`→tl,
  `argmin(y-x)`→tr, `argmax(x+y)`→br, `argmax(y-x)`→bl.
- **Why "below 45 degrees" is fragile:** this only *correctly labels* which
  physical corner is which as long as the quad's rotation relative to the
  image axes stays strictly under 45°; exactly at 45° two corners can tie on
  `sum` or `diff` (ambiguous argmin/argmax — numpy just picks the first
  occurrence, silently), and past 45° the wrong corner is picked as tl/tr/etc.
  (e.g. what is physically top-right becomes labeled top-left), which
  silently produces a **mirrored/flipped warp** — cells would still get
  extracted into an 81-cell grid, but transposed/rotated relative to the
  actual board, corrupting every downstream digit reading with no visible
  error.
- **Compounding factor — perspective skew:** the function assumes the 4 points
  approximate a rotated *rectangle*. If the quad found is a genuine
  perspective trapezoid (not just an in-plane rotation — e.g. photo taken at
  a steep camera angle rather than top-down), the sum/diff extremes can behave
  unpredictably even well under 45° rotation, because trapezoidal skew shifts
  where the coordinate extremes fall independent of rotation angle. The
  function has no explicit check that the quad looks "rectangular enough"
  before trusting sum/diff ordering.
- **Test data relevance:** `tests/create-noisy-tests.py`'s `rotate()` applies
  `angle = random.uniform(-30, 30)` (line 40) — safely under 45°, so the
  current synthetic dataset alone won't expose the boundary failure, but any
  real-world photo taken at a more extreme in-plane rotation (upside-down-ish
  book photos, etc.) would hit it silently.
- **Suggested fix:** don't rely purely on sum/diff. Compute the quad's
  centroid and use `atan2` per-point angle to order corners by angular
  position around the centroid (robust to any rotation, not just <45°), then
  separately determine which corner is "top-left" using the minimum-sum point
  *only as a tie-breaker/starting reference*, or cross-check the sum/diff
  result against the angular ordering and warn/reject when they disagree.

### 5. No explicit rotation-angle estimation/deskew — correctness depends entirely on corner-finding surviving rotation
- **Confirmed:** there is no code anywhere in this file that estimates an
  in-plane rotation angle and corrects it prior to (or instead of) the 4-point
  perspective warp. `cv2.getPerspectiveTransform` from 4 corners to a square
  is mathematically capable of undoing arbitrary rotation/perspective *if the
  4 corners are correct* — so this isn't a missing feature by itself.
- **Where rotation actually breaks things:** `_corners_from_hough`'s bucketing
  (line 211-215) sorts every Hough line into only **two** classes, "vertical"
  (`abs(theta) < pi/4`) and "horizontal" (`abs(theta - pi/2) < pi/4`), each a
  fixed 90°-wide bucket 90° apart. This hard-codes the assumption that grid
  lines are near axis-aligned; it tolerates rotation only up to just under
  ±45° before a line's `theta` falls on a bucket boundary (or, with noise
  perturbing the estimated angle, straddles it unpredictably) and gets
  silently dropped from both lists — which can push `len(horizontal) < 2` or
  `len(vertical) < 2` (line 217) and fail the whole Hough path even though the
  grid is clearly visible, just rotated near the tolerance edge.
- **Suggested fix:** derive the dominant rotation angle first (e.g. from the
  two most common Hough theta clusters, or via `cv2.minAreaRect` on the
  largest contour) and bucket lines *relative to that estimated angle* instead
  of fixed 0°/90° references — this converts the current "must be within ±45°
  of upright" constraint into "must be within ±45° of whatever angle the grid
  actually is," which is a real invariance instead of an accidental one.

### 6. The fallback stage's only actual change is removing median blur — and that change can be counterproductive for the exact noise type it's supposed to help with
- **Confirmed from code:** compare PRIMARY's recipe (`preprocess.py` lines
  30-39: CLAHE → `medianBlur(5)` → `GaussianBlur(5,5)` → adaptive threshold
  11/2) against FALLBACK's recipe (`grid_extraction.py` lines 52-56: reuse
  `pre.normalized` (CLAHE only) → `GaussianBlur(5,5)` → identical adaptive
  threshold 11/2). Canny thresholds, Hough vote threshold, contour epsilons,
  `MIN_GRID_AREA_RATIO`, and morphological close kernel are **all identical**
  between the two stages — literally the only difference is whether median
  blur ran.
- **Why this is backwards for `_sp` (salt-pepper) test images:** median blur
  is specifically the filter that removes salt-and-pepper impulse noise.
  Removing it in the fallback (as the README's own rationale for the fallback
  explains — "اگر Fallback از blurred استفاده می‌کرد... هیچ تفاوتی با Primary
  نداشت" — i.e. it exists so *something* differs) helps the case where median
  blur destroyed thin grid lines on a *clean* image, but for
  `data/bad-sodu/*_sp_*`/`*sp_gauss*`/etc. images, if PRIMARY already failed
  because salt-and-pepper noise (with median blur applied) confused contour
  detection, FALLBACK re-attempts the *same* detection on a version of the
  image with *more* residual impulse noise (no median blur at all), which is
  very unlikely to succeed and may perform strictly worse.
- **No adaptive parameter tuning based on detected noise level exists** — the
  entire 3-stage chain is fixed-threshold; there is no measurement of noise
  (e.g. Laplacian variance, salt-and-pepper pixel-fraction estimate) driving a
  choice of which recipe to try, so the fallback isn't really "a second
  strategy for noisy images," it's "the original alternate implementation that
  got kept for images where blur was the problem, not noise."
- **Suggested fix:** make the fallback noise-aware: if a cheap salt-and-pepper
  estimator (e.g. fraction of pixels at 0/255 extremes relative to local
  neighborhood) is high, fallback should apply a *larger* median filter (7 or
  9) or a bilateral filter instead of removing median blur entirely; only skip
  median blur when the estimator suggests noise is low (i.e. the PRIMARY
  failure was more likely a fine-line-erosion problem than a noise problem).

### 7. Leftover debug print in `_corners_from_hough` (line 195)
```python
def _corners_from_hough(blurred, min_area) -> np.ndarray | None:
    print("herehereedfjkahfjhfhjdhfkakj")
```
- Confirmed still present. Unconditional `print`, runs on every single call to
  the Hough fallback path (i.e. every time contour detection fails, in both
  PRIMARY and FALLBACK stages) with no gating behind a verbosity/debug flag.
  Pure leftover debug artifact — should be deleted (or replaced with a
  proper `logging.debug(...)` call if a trace point is actually wanted there).

### 8. `MIN_GRID_AREA_RATIO = 0.05` is a single fixed fraction with no per-photo calibration
- A grid photographed from farther away, or with a lot of surrounding
  background/table, can legitimately occupy less than 5% of the frame; this
  constant would reject a correctly-found quad in that case with no
  fallback other than proceeding to the next stage (which uses the *same*
  constant, so it doesn't help). Photos zoomed in past the point the ratio
  assumes 5% is achievable are also a distinct failure mode.

## Specs for future work

- **Coordinate order convention:** all corner arrays in this file, once past
  `_order_corners`, are `(tl, tr, br, bl)` — top-left, top-right,
  bottom-right, bottom-left, matching `_warp_destination()`'s
  `[[0,0],[WARP_SIZE-1,0],[WARP_SIZE-1,WARP_SIZE-1],[0,WARP_SIZE-1]]`. Any new
  code producing a 4-point array must either already be in this order or be
  passed through `_order_corners` before use with
  `cv2.getPerspectiveTransform`.
- **`pre.detect_scale` semantics:** `detect_scale <= 1.0` is the ratio
  `detect_image_size / original_image_size` used to shrink large images for
  cheaper corner detection (`preprocess.py`, `MAX_DETECT_DIM = 1024`). Corners
  found on the shrunk detect image are in *detect-image pixel coordinates*, so
  they must be **divided** by `detect_scale` (not multiplied) to convert back
  to original-image coordinates — see `grid_extraction.py` line 46-47:
  `corners = corners / pre.detect_scale`. This conversion is only applied to
  the PRIMARY stage's result; FALLBACK and LAST-RESORT already operate on
  full-resolution images/coordinates and must *not* be divided (correctly not
  done in the current code — a future change that unifies these code paths
  must preserve this asymmetry).
- **Debug image naming convention in `output_dir`:** `preprocess.py` writes
  `01_gray.png`, `02_clahe.png`, `03_denoised.png`, `04_binary.png`; this file
  continues the sequence with `05b_fallback_binary.png` (only written if
  FALLBACK actually runs), `05_grid_outline.png` (the final chosen corners
  drawn over the original image, annotated with the method name — `primary`,
  `fallback`, or `full_image_fallback`), and `06_warped.png` (the final
  `WARP_SIZE x WARP_SIZE` output). Note `05b` is written *before* `05` in
  file-listing order despite being numbered as if it were a sub-step of `05`
  — this is intentional (it's a snapshot of an intermediate fallback binary
  image, produced before the final `05_grid_outline.png` is known), but easy
  to misread as a bug when skimming `results/`. Any future stage added to the
  chain should keep using this `NN_description.png` convention and pick the
  next free number (currently nothing is written above `06`).
- **`method`/`detector` strings** (`"primary"`/`"fallback"`/
  `"full_image_fallback"` and `"contour"`/`"hough"`/`"full_image"`) are printed
  to stdout (`print(f"[grid] method used: {method} ({detector})")`) and burned
  into the `05_grid_outline.png` debug image text — if these string literals
  are ever renamed, check for any downstream log-scraping/tests that match on
  them.
- **`GridNotFoundError`** is dead code in normal operation (kept only for
  backwards compatibility per its docstring, lines 12-17) — `extract()` never
  raises it anymore due to the LAST RESORT stage. `main.py` still imports and
  catches it (`except GridNotFoundError:`), which is now unreachable dead
  code on that side too; a future cleanup pass should decide whether to keep
  the safety net or remove it from both files together.
