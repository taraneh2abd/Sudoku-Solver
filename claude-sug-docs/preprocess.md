# `src/preprocess.py` — Reference

## Purpose

`preprocess()` is the single entry point that turns a raw input image (color or
gray, `main.py` reads it with `cv2.imread`) into every derived image the rest of
the pipeline needs. It runs **once** per input image (see README.md's Persian
description, section "مرحله اول: Preprocess" — "تصویر فقط یک بار آماده‌سازی
می‌شود"). Everything downstream (`src/grid_extraction.py`) consumes its output
instead of re-deriving grayscale/blur/threshold images itself.

Two parallel pipelines are produced from the same source:

1. **Full-resolution pipeline** (`gray`, `normalized`, `blurred`, `binary`) —
   used for the actual perspective warp (`extract()` warps `pre.normalized`,
   and `_refine_corners` also operates on `pre.normalized`), so digit/line
   detail isn't lost to downsampling.
2. **Detection pipeline** (`detect_blur`, `detect_bin`, `detect_scale`) — a
   possibly-downsampled copy used only to *locate* the grid's 4 corners
   cheaply. Corners found here get divided by `detect_scale` to map back into
   full-resolution coordinates before any warping happens.

## `Preprocessed` dataclass fields

| Field | How it's built | Used downstream for |
|---|---|---|
| `gray` | `cv2.cvtColor(image, BGR2GRAY)` (or passthrough if already gray) | Source for everything else; also the fallback source for the "full image" last-resort corners (`pre.gray.shape` in `grid_extraction.extract()`) |
| `normalized` | `CLAHE(clipLimit=2.0, tileGridSize=8x8)` applied to `gray` | **This is what actually gets warped** (`cv2.warpPerspective(pre.normalized, matrix, ...)` in `extract()`), and what `_refine_corners` re-binarizes to find sharper corners. It is also the base image for the FALLBACK detection stage in `grid_extraction.py` |
| `blurred` | `GaussianBlur(medianBlur(normalized, 5), (5,5), 0)` | Only used to build `binary`; not directly consumed elsewhere in the current code (kept for API completeness / debug dump `03_denoised.png`) |
| `binary` | `adaptiveThreshold(blurred, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2)` | Not used directly downstream in the current code path — the PRIMARY grid search uses `detect_bin`, not `binary`, unless `detect_scale == 1.0` in which case `detect_bin is binary` (see below). Saved to disk as `04_binary.png` for debugging |
| `detect_blur` | Same recipe as `blurred` but computed on a resized `detect_gray` when downsampling kicks in; otherwise **is the same object as** `blurred` | Passed to `_locate_grid` as the Hough-fallback source image (edge detection input) |
| `detect_bin` | Same recipe as `binary` but on the resized image; otherwise **is the same object as** `binary` | Passed to `_locate_grid` as the contour-search source image |
| `detect_scale` | `min(1.0, MAX_DETECT_DIM / max(gray.shape[:2]))` | Corners found on the (possibly) downsampled `detect_bin`/`detect_blur` are divided by this scale in `extract()` (`corners = corners / pre.detect_scale`) to map back to full-resolution pixel coordinates. When the image is already `<= MAX_DETECT_DIM`, `detect_scale == 1.0` and no resize/division happens |

Note the aliasing: when `max(gray.shape[:2]) <= MAX_DETECT_DIM`, `detect_blur`
and `detect_bin` are literally the *same* arrays as `blurred`/`binary` (line
53-54: `detect_blur = blurred; detect_bin = binary`), not copies. Anything that
mutates one in place would silently mutate the other — currently nothing does,
but a future edit that adds in-place mutation to either path needs to be aware
of this aliasing.

## Key constants

- `MAX_DETECT_DIM = 1024` — the downsample cap for the detection pipeline. The
  inline comment ("block size 11 is good for 500–1024px images") documents a
  coupling: the adaptive-threshold `blockSize=11` used both here and in
  `grid_extraction.py`'s fallback path was tuned assuming the detection image
  lands in the 500–1024px range. If `MAX_DETECT_DIM` is changed without
  revisiting `blockSize`, the tuning assumption silently breaks.
- Adaptive threshold params `(blockSize=11, C=2)` — hardcoded in two places in
  this file (full-res `binary` and downsampled `detect_bin`), and duplicated a
  third time in `grid_extraction.py`'s fallback stage. No single source of
  truth; changing one requires remembering to change the others (or extracting
  a shared constant/function).
- CLAHE params `clipLimit=2.0, tileGridSize=(8,8)` — fixed regardless of input
  image size or measured contrast; see `docs/grid_extraction.md` for how this
  interacts with the synthetic "shadow" test images (`data/bad-sodu/*_shadow*`).

## Debug output (`output_dir`)

When `output_dir` is passed, `preprocess()` writes://
- `01_gray.png`
- `02_clahe.png`
- `03_denoised.png` (actually the Gaussian+median blurred image, i.e. `blurred`, despite the "denoised" name suggesting it's just the median-blur step)
- `04_binary.png`

These numbers continue in `grid_extraction.py` (`05_*`, `06_*`), so the two
modules share one implicit numbering convention for `results/` debug images —
see `docs/grid_extraction.md`'s "Specs for future work" section.

## Issues noted

1. **No noise-adaptive behavior at all.** CLAHE clip limit, median blur kernel
   (5), Gaussian kernel (5,5), and adaptive-threshold block size/C (11, 2) are
   all fixed regardless of measured image noise/contrast. There is no
   auto-detection of noise level (e.g. via local variance or a noise
   estimator) to decide whether median blur radius should increase for heavy
   salt-and-pepper corruption, or whether CLAHE clip limit should increase for
   strongly shadowed images.
2. **`detect_bin`/`detect_blur` aliasing** (see above) is a subtle sharp edge
   for future maintainers — it's not a bug today but is easy to turn into one.
3. **Duplicated magic numbers** — `(11, 2)` for adaptive threshold appears 3
   times across `preprocess.py` and `grid_extraction.py` with no shared
   constant, and CLAHE params aren't reused/shared between the full-res and
   detect-scale branches via a helper (the recipe is copy-pasted in the `if
   detect_scale < 1.0` branch).
4. **`binary` is effectively unused** in the current pipeline once
   `detect_scale < 1.0` (only `detect_bin`/`detect_blur` feed `_locate_grid`);
   it exists purely for the debug dump and API completeness, which isn't
   obvious from the field alone — worth a comment in the dataclass if this is
   intentional.
