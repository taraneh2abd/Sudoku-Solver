# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Section docs (read before making non-trivial changes)

Detailed per-module reference docs — architecture, data flow, constants, known bugs with
concrete evidence/fixes, and "specs for future work" — live in `docs/`. Start at
**[docs/README.md](docs/README.md)**, which indexes:

- `docs/preprocess.md` — `src/preprocess.py`
- `docs/grid_extraction.md` — `src/grid_extraction.py` (grid-corner detection/warp)
- `docs/cell_extraction.md` — `src/cell_extraction.py` (cell cropping/cleanup)
- `docs/digit_recognition.md` — `src/digit_model.py`, `src/digit_recognizer.py`, `src/train.py`, `test.py`
- `docs/solver_and_app.md` — `src/solve_sudoku.py`, `app.py`, `main.py`, the frontend

`docs/README.md` also has a "Known bugs — status" section. As of 2026-07-08, everything on
that list is fixed except the digit-recognition model/checkpoint mismatch (left alone
deliberately — fixing it requires running/comparing models). Check it before assuming a
symptom is new — it may already be root-caused (and fixed, or intentionally not) there.

## Commands

There is no test suite, linter, or build step configured in this repo (no pytest/tox/lint
config exists — `tests/` contains a manual fixture-generation script, not automated tests).

```bash
# install deps
pip install -r requirements.txt

# run the full pipeline on one image (writes debug artifacts to results/, wipes that dir first)
python main.py path/to/image.jpg

# optional second arg: write to a specific output dir instead of results/
# (this is what app.py uses so concurrent uploads don't share one output dir)
python main.py path/to/image.jpg results/some_job_id

# run the Flask UI (calls `python main.py` as a subprocess per upload, see docs/solver_and_app.md)
python app.py

# retrain the digit classifier (SudokuDigitCNN, src/digit_model.py) — manual, not in the pipeline
python -m src.train --data-dir data/processed/digits --epochs 10 --output models/new_ds_10_epoch.pt

# regenerate the training dataset from the Lexski HF dataset — manual, one-off, not in the pipeline
python generate_dataset.py --output data/processed/digits
```

## Architecture

Single-image CLI pipeline (`main.py`), wrapped by a Flask UI (`app.py`) that shells out to
`main.py` per upload and polls for JSON output files rather than calling into the pipeline
in-process:

```
image -> preprocess() -> extract() -> save_cells() -> DigitRecognizer.predict_board() -> solve()
         (preprocess.py)  (grid_extraction.py) (cell_extraction.py)  (digit_recognizer.py)  (solve_sudoku.py)
```

- **`preprocess.py`** builds several grayscale/CLAHE/blur/threshold variants of the input image
  once (`Preprocessed` dataclass), including a separately-scaled "detect" variant capped at
  `MAX_DETECT_DIM` for grid-finding.
- **`grid_extraction.py`** locates the board's 4 corners via a 3-stage strategy — primary
  (contour → Hough on the downsampled detect image, auto-Canny thresholds), fallback (same
  search with a lighter median blur, full resolution), full-image last resort (never
  hard-fails) — then perspective-warps to a fixed `WARP_SIZE x WARP_SIZE` (450) image. Corner
  ordering (`_order_corners`) sorts by angle around the quad's centroid, which stays correct
  for any rotation (not just below ~45°, as the old sum/diff heuristic was). `GridNotFoundError`
  is defined but no longer raised by `extract()`; the except-branch in `main.py` is kept as a
  harmless defensive fallback even though it's currently unreachable.
- **`cell_extraction.py`** slices the warped image into 81 `CELL_SIZE x CELL_SIZE` cells
  (`CELL_SIZE = WARP_SIZE // 9`), decides empty-vs-filled per cell, and cleans filled cells
  (Otsu threshold, inverse-detection, Hough-line removal near edges, connected-component noise
  filtering). **`generate_dataset.py` (repo root) reuses `make_raw_cell`/`process_cell` directly**
  to build the training set from the Lexski HF dataset — this file is on the critical path for
  both inference and training-data generation; a fix applied here without regenerating
  `data/processed/digits` creates a train/inference mismatch.
- **`digit_recognizer.py`** loads a model and classifies each non-empty 28x28 cell. Note: the
  *active* code path loads a whole pickled `LeNet` from `models/mnist_lenet.pt`; this is a
  separate lineage from `src/digit_model.py`'s `SudokuDigitCNN` + `src/train.py`, whose
  state_dict checkpoints (`models/new_ds*.pt`, `models/old_black_white_ds*.pt`) are not
  currently loaded by anything at inference time. See `docs/digit_recognition.md` before
  swapping models — per the previous author's own note, doing so touches `main.py`,
  `digit_model.py`, `digit_recognizer.py`, and `train.py` together (no single loading
  abstraction exists).
- **`solve_sudoku.py`** is a plain recursive backtracking solver (no heuristics/constraint
  propagation) operating on a 9x9 list-of-lists board (`0` = empty).
  `board_has_conflicts(board)` checks the givens for a row/column/box duplicate before
  solving, so a genuinely unsolvable board can be told apart from one that only looks
  unsolvable because the digit recognizer misread a cell.
- **`app.py`** accepts an upload, assigns it a `uuid4` job id, writes it to `app/uploads/`
  under a job-id-prefixed filename, launches `python main.py <path> results/<job_id>` as a
  subprocess, then polls the filesystem for `<job_dir>/00_original.json` and
  `<job_dir>/00_solved.json`, each under its own fresh 30s timeout. If the subprocess exits
  early without producing the expected file, its stderr is surfaced immediately instead of
  waiting out the timeout. The solved-board JSON is a 9x9 array, the literal string
  `"UNSOLVABLE"` (no solution exists), or `"INVALID_BOARD"` (the givens themselves conflict —
  likely an OCR misread) — `app/static/script.js` branches on these. The app has no other
  coupling to the CV pipeline: get the JSON shape right and the app works, per the original
  author's note in the (Persian) README.

## Dataset/model provenance

- Training images come from the `Lexski/sudoku-image-recognition` HuggingFace dataset via
  `generate_dataset.py`, which also encodes the label semantics for "empty" vs "digit" vs
  "ambiguous/multi-candidate" cells (`cell_label()`) — this is the ground-truth definition of
  cell emptiness for training, and it currently disagrees with the raw-stddev heuristic
  `cell_extraction.make_raw_cell` uses at inference time (see `docs/cell_extraction.md`).
- `test.py` (repo root) is orphaned scratch code — a third/fourth model architecture, a
  hardcoded personal path, and a reference to a model file that doesn't exist in `models/`. It
  is not part of the maintained pipeline.
