# `src/cell_extraction.py` — Cell Extraction Pipeline

## Fix log (2026-07-08)

All 3 bugs below were fixed directly in `src/cell_extraction.py` (and one call site in
`generate_dataset.py` updated to match a changed function signature):

- **Bug 1 (`EDGE_RATIO`)**: changed `EDGE_RATIO = 0.0` -> `0.15`, restoring the intended
  Hough-line-removal margin.
- **Bug 2 (multi-candidate cells)**: `clean_noise()` now returns `(cleaned, kept)` — the
  count of connected components that survived — and `process_cell()` propagates that tuple.
  `save_cells()` now treats `kept > MAX_DIGIT_COMPONENTS` (`MAX_DIGIT_COMPONENTS = 1`) as
  `cell.is_empty = True` instead of handing a multi-blob image to the classifier.
  `generate_dataset.py`'s call site was updated to unpack the new tuple return
  (`processed, _kept = process_cell(...)`); its label semantics (`cell_label`) were left
  unchanged since that's a training-data/model-retraining decision, out of scope here.
- **Bug 3 (asymmetric margin)**: `make_raw_cell()` now computes `margin_y`/`margin_x` once
  from the original cell span before mutating `y0`/`x0`, restoring a symmetric 6px/6px crop.

Verified without running any model: `python -m py_compile`, then ran the full
`preprocess -> extract -> save_cells` chain (no digit classifier involved) against several
`data/bad-sodu/*.jpg` noisy/rotated fixtures to confirm no crashes and sane empty/filled
cell counts.

## Fix log (2026-07-08, follow-up) — shadow artifact false-triggering the multi-candidate check

**Symptom:** a real photographed board (handwritten, candidate-style puzzle) had cell `r1c6`
(a single bold "2") wrongly treated as empty. Root cause traced with
`cv2.connectedComponentsWithStats` on the actual failing cell: Otsu thresholding under an
uneven shadow left a small corner artifact (area 20px, bbox touching two crop edges at once)
alongside the real "2" stroke (area 59px). That gave `clean_noise` a component count of 2,
which the Bug-2 fix above (correctly) reads as "multi-candidate -> empty" — but this cell only
had one real digit; the second "component" was shadow noise, not a pencil mark.

**Fix:** added `MIN_COMPONENT_AREA_RATIO = 0.5` to `clean_noise`. A component only counts
towards `kept` if its area is at least half the area of the largest surviving component in
that cell. Rationale: a genuine multi-candidate cell holds several pencil marks of comparable
size to each other (no single dominant blob), while a shadow/threshold artifact next to one
real digit is disproportionately small relative to that digit. Confirmed on the actual
failing image (`results/06_warped.png` from that run): `r1c6` area ratio was 20/59 = 0.34,
now correctly excluded (`kept` drops from 2 to 1).

**Validated for regressions**, all on the same real board (no model involved):
- Re-ran all 81 cells before/after the fix: 9 cells flipped from "false multi-candidate" to
  correctly "single digit" (`r0c6`, `r1c6`, `r3c0`, `r4c3`, `r5c2`, `r6c4`, `r7c0`, `r7c3`,
  `r8c1`); visually inspected several of these (`r5c2`, `r7c0`) and confirmed each is a single
  bold digit with a small stray fragment, not a real second candidate.
- Spot-checked cells that remained flagged multi-candidate after the fix (`r0c1`, `r0c2`,
  `r2c1`) and visually confirmed each genuinely contains 2+ comparably-sized pencil-mark
  digits — the fix did not suppress real multi-candidate detection.

## Purpose

After `src/grid_extraction.py` perspective-warps the detected Sudoku board into a
`WARP_SIZE x WARP_SIZE` (450x450) grayscale image, `cell_extraction.py` is responsible for:

1. Slicing that warped image into an ordered 9x9 grid of 81 cell images.
2. Deciding, per cell, whether it is empty or contains a digit.
3. Cleaning up non-empty cells (binarize, remove grid-line bleed, remove small noise blobs)
   so that only a digit-shaped blob remains, ready to be resized to 28x28 and classified by
   `src/digit_recognizer.py`.

The same two building-block functions (`make_raw_cell`, `process_cell`) are reused by
`generate_dataset.py` (repo root) to turn the Lexski HuggingFace dataset into the training
images for the digit classifier. **This module is therefore on the critical path for both
inference and training-data generation** — any change to cropping/thresholding here changes
what the model is trained on, not just what it sees at inference time.

## Data flow

```
warped (450x450 grayscale, from grid_extraction.WARP_SIZE)
  │
  ▼
save_cells(warped, output_dir)
  │
  ├─ for each (row, col) in 9x9:
  │     make_raw_cell(warped, row, col) -> Cell(image=raw_crop, is_empty=bool)
  │        - slice CELL_SIZE x CELL_SIZE block
  │        - shrink by CELL_MARGIN_RATIO on each side (see Bug 3 — asymmetric in practice)
  │        - is_empty = std(raw_crop) < 8.0   (decided BEFORE any thresholding)
  │
  ├─ for each non-empty cell:
  │     process_cell(cell.image) -> cleaned binary image
  │        1. Otsu threshold -> binary (0/255)
  │        2. if white_ratio > WHITE_RATIO_FOR_INVERSE (0.55): invert
  │           (handles cells that binarized to a mostly-white/inverted appearance)
  │        3. HoughLinesP to find near-horizontal/near-vertical line segments,
  │           build a removal mask for segments within EDGE_RATIO of the crop's edges,
  │           subtract that mask from the binary image
  │        4. clean_noise(): connectedComponentsWithStats(connectivity=8), keep only
  │           components with area >= MIN_NOISE_AREA (15), drop everything else
  │
  └─ write non-empty cells' processed images to <output_dir>/cells/cell_r{row}c{col}.png
     return the full list of 81 Cell objects (empty cells keep their *raw*, unprocessed
     image — save_cells never overwrites cell.image for is_empty cells)
  │
  ▼
digit_recognizer.DigitRecognizer.predict_board(cells)
  - is_empty cells -> board value 0, no model call
  - non-empty cells -> cv2.resize to 28x28 -> torchvision transform -> LeNet -> argmax digit
```

`generate_dataset.py::generate_split` calls `make_raw_cell` + `process_cell` directly (not
`save_cells`), and instead of `is_empty`, the *label* for each cell comes from
`cell_label(flags)`:

- `flags[0]` = "solved" bit. `flags[1:10]` = which of digits 1-9 are marked as pencil-mark
  candidates in that cell.
- Not solved + zero candidate flags set -> label `0` (genuinely blank cell).
- Not solved + one-or-more candidate flags set -> label `None` -> **that cell is skipped
  entirely** and never written to the training set (`generate_split` does
  `if label is None: continue`).
- Solved + exactly one candidate flag set -> label = that digit.
- Solved + zero or 2+ flags set -> label `None` -> skipped (inconsistent/ambiguous data).

So the training data **never contains an example of a multi-candidate (pencil-marked, not
yet solved) cell** — those rows are dropped at dataset-generation time, not remapped to the
"empty" class. See Bug 2 below for why this matters at inference time.

## Constants

| Constant | Value | Meaning |
|---|---|---|
| `CELL_SIZE` | `WARP_SIZE // 9` = 50 | Pixel size of one grid cell in the warped 450x450 image. |
| `CELL_MARGIN_RATIO` | 0.12 | Fraction of each cell's raw span trimmed off each side in `make_raw_cell`, before any processing, to try to cut off the board's grid lines. Intended ~6px margin on a 50px cell. |
| `WHITE_RATIO_FOR_INVERSE` | 0.55 | If more than 55% of the Otsu-thresholded binary cell is white, the cell is inverted. Assumes digit ink should be the minority-color foreground. |
| `MIN_NOISE_AREA` | 15 | Minimum connected-component pixel area to survive `clean_noise`. Anything smaller is treated as noise and zeroed out. |
| `EDGE_RATIO` | 0.15 (was 0.0) | How close to the cell edge a detected line segment must be to be considered a grid-line remnant and removed (see Bug 1 — fixed). |
| `MAX_DIGIT_COMPONENTS` | 1 | Max components allowed before a cell is treated as multi-candidate/empty (see Bug 2 — fixed). |
| `MIN_COMPONENT_AREA_RATIO` | 0.5 | A component only counts towards the multi-candidate check if its area is at least this fraction of the largest component's area in that cell — filters small shadow/threshold artifacts from being mistaken for a second candidate digit (see the shadow-artifact follow-up fix below). |

## `Cell` dataclass contract

```python
@dataclass
class Cell:
    image: np.ndarray   # raw crop if is_empty, else the cleaned binary image after process_cell
    is_empty: bool       # decided once in make_raw_cell from the RAW crop's stddev
```

Consumers (`digit_recognizer.DigitRecognizer.predict_board`) rely on `is_empty` to skip
model inference entirely (`board.append(0)`), and on `cell.image` being either a raw grayscale
crop (empty) or an 8-bit binary (0/255) image containing (ideally) just the digit stroke
(non-empty). `cv2.resize(cell.image, (28, 28))` is called downstream without any
letterboxing/aspect-preservation — the crop's aspect ratio (affected by Bug 3's asymmetry)
is stretched directly to 28x28.

---

## Known Bugs

### Bug 1 — `EDGE_RATIO = 0.0` disables the grid-line-removal step, so lines can bleed into and corrupt digit content

**Root cause:** In `process_cell`:

```python
EDGE_RATIO = 0.0
...
margin_x = int(w * EDGE_RATIO)   # = 0
margin_y = int(h * EDGE_RATIO)   # = 0
...
if x <= margin_x or x >= w - margin_x:   # x <= 0  or  x >= w
```

With `margin_x = margin_y = 0`, the "near the edge" test collapses to "the line's midpoint
x-coordinate is exactly pixel 0" (for near-vertical lines) or "exactly pixel 0" for `y`
(near-horizontal lines). Since pixel coordinates in a `w`-wide image run `0..w-1`, `x >= w`
is essentially unreachable, so in practice **only a line sitting exactly on the crop's
leftmost/topmost pixel column/row gets removed.** Any grid-line remnant that survived the
`CELL_MARGIN_RATIO` pre-crop by even 1px will not be caught by this Hough-line removal logic
at all — the loop runs, `cv2.HoughLinesP` may detect the line, but the `if` guard almost
never passes, so `remove_mask` stays (near) empty and `binary` is returned essentially
unmodified with respect to line content.

**Concrete evidence:** `tests/cell.png` / `tests/result.png` (the file's own committed
before/after example, produced by the commented-out driver code at the bottom of the file)
show a thin vertical bar surviving on the left edge of the digit "3" in *both* the raw and
the "processed" result image — i.e. the line-removal step visibly did not remove it.

**Why the pre-crop margin doesn't save you:** `CELL_MARGIN_RATIO = 0.12` on `CELL_SIZE = 50`
trims only `int(50 * 0.12) = 6` px per side (5px on the bottom/right — see Bug 3). Sudoku
grid lines in a perspective-warped 450x450 image are frequently thicker than 6px near the
board's outer border (perspective foreshortening/anti-aliasing widens them), and warp
misalignment can shift a line a few pixels off from where a naive fixed-ratio crop expects
it. So a 5-6px margin is not reliably enough to fully exclude the line, and the one
mechanism designed to clean up whatever survives (the Hough-line removal in `process_cell`)
is disabled by `EDGE_RATIO = 0.0`.

**Impact:** Line remnants that are >= `MIN_NOISE_AREA` (15px) survive `clean_noise` (which
only filters by *size*, not by *shape* or *position*) and either (a) get treated as part of
the digit's connected component if they touch it, distorting the digit's shape before the
28x28 resize and confusing the classifier, or (b) remain as a separate surviving blob,
which for a digit cell adds spurious ink that a `cv2.resize` will blend into the digit
region.

**Suggested fix:** Restore a real edge fraction, e.g. `EDGE_RATIO = 0.15` (roughly matching
the pre-crop margin's proportions on the *already-cropped* ~38-39px cell, i.e. ~5-6px), and/or
fix the boundary comparison range (`x >= w - margin_x` is fine once `margin_x > 0`; the bug is
purely that `margin_x` evaluates to 0). Minimal one-line fix:

```python
EDGE_RATIO = 0.15   # was 0.0
```

### Bug 2 — Multi-candidate (pencil-mark) cells are never detected and are fed to the single-digit classifier as if they contained one solved digit

**Root cause:** `is_empty` is decided once, in `make_raw_cell`, purely from
`np.std(crop) < 8.0` on the **raw, unprocessed** grayscale crop (line 81:
`is_empty = float(np.std(crop)) < 8.0`). A cell with several small pencil-mark digits has
much more pixel variance than a truly blank cell, so `is_empty` evaluates to `False` and the
cell is routed through `process_cell` exactly like a normal solved-digit cell.

`process_cell` -> `clean_noise` (`connectedComponentsWithStats(connectivity=8)`) keeps
**every** component with `area >= MIN_NOISE_AREA (15)` — there is no cap on how many
components survive, and no logic anywhere in the file that counts the resulting components
and reasons "more than one digit-shaped blob here -> this must be a multi-candidate cell,
treat it as empty/ambiguous instead of running the classifier." I read `clean_noise` and
`save_cells` fully and confirmed: `clean_noise` (lines 138-150) unconditionally unions all
qualifying components into `cleaned`; `save_cells` (lines 49-51) only ever branches on the
already-decided `cell.is_empty`, never re-examines `cell.image` after processing. So a cell
with, say, 4 pencil-marked candidate digits is cropped, binarized, "cleaned" (all 4 marks
kept if each is >=15px), resized to a single 28x28 image, and handed to a single-class LeNet
softmax — producing one arbitrary/garbage digit prediction for what should be an empty cell.

**Why the model was never trained to do the right thing here either:**
`generate_dataset.py::cell_label` (lines 92-118) explicitly returns `None` — meaning "skip,
do not write this training example" — for any not-yet-solved cell with 1+ candidate flags
set (`len(digits) == 0: return 0`, else for `1 <= len(digits)`: falls through to `return
None` since the `solved` branch never applies). Only a cell with **zero** candidate flags
becomes label `0`. So the multi-candidate case is entirely absent from the training set —
the model has no learned representation for "this is a pencil-marked, not-yet-solved cell,"
because `generate_split` never wrote such an example to disk for it to train on.

**Impact:** Any real photographed puzzle that has pencil marks (a very common real-world
case for "in-progress" Sudoku photos) will have candidate cells misread as solved digits,
producing wrong board values and downstream solver failures/incorrect solves.

**Suggested fix:** After `clean_noise`, count the surviving components and treat >1
significant component as "not a single solved digit":

```python
def clean_noise(cell_image) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cell_image, connectivity=8)
    cleaned = np.zeros_like(cell_image)
    kept = 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= MIN_NOISE_AREA:
            cleaned[labels == label] = 255
            kept += 1
    return cleaned, kept
```

and in `save_cells`, treat `kept > 1` as `is_empty = True` (or a new `Cell.is_ambiguous`
flag so callers can decide to render "?" instead of "0"), e.g.:

```python
cleaned, kept = clean_noise(binary_after_line_removal)
if kept > 1:
    cell.is_empty = True   # multiple candidate marks, not a single solved digit
else:
    cell.image = cleaned
```

This also requires a matching change in `generate_dataset.py` if the team ever wants the
model itself to learn to output "empty" for multi-candidate cells: relabel those `None`
cases as `0` instead of skipping them, so the classifier actually sees such examples during
training. As-is, doc-only note: fixing `clean_noise`'s component count without also changing
`cell_label` only fixes the *empty-cell routing*, not the model's training distribution.

### Bug 3 — Asymmetric margin in `make_raw_cell`: bottom/right margin is 1px smaller than top/left margin because `(y1 - y0)` is recomputed from the already-mutated `y0`

**Root cause:**

```python
y0, y1 = row * CELL_SIZE, (row + 1) * CELL_SIZE      # e.g. row=0: y0=0, y1=50
x0, x1 = col * CELL_SIZE, (col + 1) * CELL_SIZE

y0 = y0 + int((y1 - y0) * CELL_MARGIN_RATIO)          # int(50*0.12)=6  -> y0 = 6
y1 = y1 - int((y1 - y0) * CELL_MARGIN_RATIO)          # (y1-y0) is now (50-6)=44, not 50
                                                        # int(44*0.12)=5  -> y1 = 50-5 = 45
x0 = x0 + int((x1 - x0) * CELL_MARGIN_RATIO)          # same pattern
x1 = x1 - int((x1 - x0) * CELL_MARGIN_RATIO)
```

**Exact numbers (CELL_SIZE=50, CELL_MARGIN_RATIO=0.12):**

- Intended (symmetric) behavior: trim `int(50 * 0.12) = 6` px off *both* the top/left and
  the bottom/right -> crop span = `50 - 6 - 6 = 38` px.
- Actual behavior: top/left margin = `int(50 * 0.12) = 6` px (unaffected, computed from the
  original span). Bottom/right margin = `int((50 - 6) * 0.12) = int(44 * 0.12) = int(5.28) =
  5` px, because the second line's `(y1 - y0)` (respectively `(x1 - x0)`) is evaluated
  **after** `y0` (`x0`) was already advanced by 6, so the span used is 44 instead of 50.
- Net result: crop span = `50 - 6 - 5 = 39` px (one pixel taller/wider than intended), with
  the extra pixel sitting on the **bottom/right** side of every cell. This is a systematic,
  reproducible 1px asymmetry, not rounding noise — the bottom/right edge of every single one
  of the 81 cells keeps 1 more raw pixel than the top/left edge.

**Compounding with Bug 1:** Because the bottom/right margin is thinner (5px vs. the intended
6px), a grid line is *more* likely to have a sliver survive into the crop on that side. Since
`EDGE_RATIO = 0.0` means `process_cell`'s Hough-line removal essentially never fires (Bug 1),
whatever extra sliver of grid line leaks in on the bottom/right due to this asymmetry has no
downstream mechanism left to remove it before `clean_noise`, if it forms/joins a component
>= 15px.

**Suggested fix:** Compute the margin once, from the original span, and reuse it for both
edges:

```python
def make_raw_cell(warped, row, col) -> Cell:
    y0, y1 = row * CELL_SIZE, (row + 1) * CELL_SIZE
    x0, x1 = col * CELL_SIZE, (col + 1) * CELL_SIZE

    margin_y = int((y1 - y0) * CELL_MARGIN_RATIO)
    margin_x = int((x1 - x0) * CELL_MARGIN_RATIO)

    y0, y1 = y0 + margin_y, y1 - margin_y
    x0, x1 = x0 + margin_x, x1 - margin_x

    crop = warped[y0:y1, x0:x1]
    is_empty = float(np.std(crop)) < 8.0
    return Cell(crop, is_empty)
```

This restores the intended symmetric 6px/6px margin (38x38 crop) on both axes.

---

## Specs for future work

Before touching this file, a future engineer needs to know:

1. **`Cell` dataclass contract.** `image` is either a *raw* grayscale crop (if `is_empty`)
   or a *cleaned binary (0/255)* image (if not). Nothing in this file guarantees a fixed
   output size for `cell.image` before the 28x28 resize downstream — crop size depends on
   `CELL_SIZE` and the (currently asymmetric, see Bug 3) margin math, so it varies by ~1px
   between top/left and bottom/right of a cell but is otherwise constant across all 81
   cells for a given `WARP_SIZE`.

2. **Coordinate/margin conventions.** Cells are addressed `(row, col)` with `row` = vertical
   index (`y`), `col` = horizontal index (`x`), consistent with `warped[y0:y1, x0:x1]`
   slicing (NumPy row-major). `CELL_MARGIN_RATIO` is applied *before* any thresholding, on
   the raw grayscale image, specifically to cut off the board's printed grid lines before
   they can be mistaken for digit ink. `EDGE_RATIO` is a **second**, independent
   line-removal mechanism that operates *after* thresholding, using Hough line detection —
   it is not just a duplicate of `CELL_MARGIN_RATIO`, it is meant to catch whatever the
   pre-crop margin missed. Bug 1 makes this second mechanism a no-op; do not assume it is
   doing anything until `EDGE_RATIO` is fixed.

3. **`WARP_SIZE` dependency.** `CELL_SIZE = WARP_SIZE // 9` is imported from
   `src/grid_extraction.py` (`WARP_SIZE = 450`). If `WARP_SIZE` ever changes,
   `CELL_SIZE` changes automatically, but `generate_dataset.py` has its **own**,
   independently hard-coded `BOARD_SIZE = 450` / `CELL_SIZE = BOARD_SIZE // 9` — these two
   constants are not derived from a single source of truth. **Changing `WARP_SIZE` in
   `grid_extraction.py` without also updating `generate_dataset.py`'s `BOARD_SIZE` will
   silently desync inference-time cell geometry from training-time cell geometry**, since
   the classifier would then be trained on crops from a different absolute pixel grid than
   what it sees at inference.

4. **`generate_dataset.py` reuses `make_raw_cell`/`process_cell` directly, bypassing
   `save_cells`/`is_empty`.** Any change to the cropping or line-removal logic in this file
   changes what the *training images* look like the next time someone regenerates the
   dataset (`python generate_dataset.py`), not just what the live app sees. A fix applied
   only at inference time (e.g. patching `EDGE_RATIO`) without regenerating the dataset
   will create a train/inference mismatch; ideally fix here and regenerate
   `data/processed/digits` before retraining.

5. **`cell_label` (`generate_dataset.py`) semantics are the source of truth for what
   "empty" means in training data**, and they currently **do not agree** with what
   `make_raw_cell.is_empty` computes at inference time: `cell_label` uses the dataset's own
   ground-truth `flags` (structured, not vision-derived) and explicitly **skips** (does not
   write to disk) any not-yet-solved cell with 1+ candidate flags, rather than labeling it
   `0`. `make_raw_cell.is_empty`, by contrast, is a raw-pixel heuristic (`std < 8.0`) that
   has no access to "was this solved" ground truth and cannot distinguish "blank" from
   "has faint pencil marks" from "has one clear digit." Any future fix to Bug 2 needs to
   decide, and document, how these two independent "empty" definitions should relate (see
   Bug 2's suggested fix for one concrete proposal: relabel multi-candidate `None` cases in
   `cell_label` as `0` if `clean_noise` is also changed to route multi-blob cells to
   `is_empty = True`).

6. **`connectivity=8` in `clean_noise`.** 8-connectivity treats diagonally-adjacent
   foreground pixels as part of the same component (as opposed to 4-connectivity, which
   only counts orthogonal neighbors). This matters for anti-aliased/thin digit strokes,
   which can be diagonally-thin at corners — switching to `connectivity=4` would fragment a
   single digit into multiple small components, likely dropping below `MIN_NOISE_AREA` and
   corrupting more digits than it fixes. Any future "count components to detect
   multi-candidate cells" fix (Bug 2) needs to keep `connectivity=8` for this reason, or the
   component count itself becomes unreliable as a signal.

7. **`MIN_NOISE_AREA = 15` and `WHITE_RATIO_FOR_INVERSE = 0.55`** were evidently tuned by
   eyeballing examples like `tests/cell.png`/`tests/result.png` — there is no automated test
   asserting behavior on a labeled corpus of noisy cells. `tests/create-noisy-tests.py`
   generates synthetic noisy variants (salt-and-pepper, Gaussian noise, rotation, shadow) of
   images in `data/bad-sodu/`, applying 2 random degradations per image — useful for manual
   spot-checking `process_cell`'s robustness, but it does not assert correctness
   automatically (no expected-output comparison, just writes new files to disk with `_sp`,
   `_gauss`, `_rot`, `_shadow` suffixes). Any future constant tuning should ideally add a
   real regression test using outputs like these as fixtures.
