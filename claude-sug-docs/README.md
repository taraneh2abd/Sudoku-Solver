# Sudoku-Solver — Section Index

Reference docs for each stage of the pipeline. Each file covers: purpose, data flow,
constants, a **Known Bugs/Issues** section (root cause + evidence + concrete fix), and a
**Specs for future work** section (what you must know before touching that file).

Pipeline order:

```
image -> preprocess.py -> grid_extraction.py -> cell_extraction.py -> digit_recognizer.py -> solve_sudoku.py
                                                                                                    |
                                                                                              app.py (Flask UI)
```

| Doc | Covers | Read this before touching... |
|---|---|---|
| [preprocess.md](preprocess.md) | `src/preprocess.py` | Gray/CLAHE/blur/threshold variants, the `Preprocessed` dataclass |
| [grid_extraction.md](grid_extraction.md) | `src/grid_extraction.py` | Grid-corner detection (primary/fallback/full-image), perspective warp, corner ordering |
| [cell_extraction.md](cell_extraction.md) | `src/cell_extraction.py`, and the parts of `generate_dataset.py` that reuse it | Cell cropping, empty-cell detection, grid-line removal, noise cleanup |
| [digit_recognition.md](digit_recognition.md) | `src/digit_model.py`, `src/digit_recognizer.py`, `src/train.py`, `test.py` | Model architecture, checkpoint format, which `models/*.pt` is actually loaded |
| [solver_and_app.md](solver_and_app.md) | `src/solve_sudoku.py`, `app.py`, `main.py`, `app/templates`, `app/static` | Backtracking solver, Flask upload/poll flow, the JSON handoff contract |

## Known bugs — status (last updated 2026-07-08)

Full details, evidence, and fixes are in the linked docs' "Fix log" sections — this is just a
pointer. Everything except the digit-recognition model mismatch was fixed on 2026-07-08
without running or loading any model (verified via `py_compile` plus direct calls into
`preprocess`/`grid_extraction`/`cell_extraction`/`solve_sudoku` on real fixture images, and
dummy-subprocess tests for `app.py`'s polling logic).

- ✅ **FIXED** — Grid detection robustness ([grid_extraction.md](grid_extraction.md)): auto-Canny
  thresholds (median-based) replace the fixed (50,150) pair; the fallback stage now uses a
  lighter `medianBlur(3)` instead of dropping median blur entirely (which hurt salt-and-pepper
  robustness); `_order_corners` now sorts by angle around the quad's centroid, correct for any
  rotation instead of only below ~45°; leftover debug `print(...)` removed.
- ✅ **FIXED** — Grid lines bleeding into cells ([cell_extraction.md](cell_extraction.md) Bug 1):
  `EDGE_RATIO` restored from `0.0` to `0.15`.
- ✅ **FIXED** — Multi-candidate (pencil-mark) cells not detected as empty
  ([cell_extraction.md](cell_extraction.md) Bug 2): `clean_noise`/`process_cell` now report how
  many digit-shaped components survived; `save_cells` marks a cell empty when more than one
  does, instead of feeding a multi-digit blob to the classifier.
- ✅ **FIXED (follow-up)** — shadow-induced Otsu artifact false-triggering the above check
  ([cell_extraction.md](cell_extraction.md), "shadow artifact" fix log): a shadow could leave a
  small extra blob after thresholding, wrongly counted as a second candidate and flipping a
  real single-digit cell to empty. `clean_noise` now only counts a component if its area is at
  least half (`MIN_COMPONENT_AREA_RATIO = 0.5`) of the cell's largest component — found and
  fixed against a real failing case (cell r1c6 of a user-provided photo), validated against
  regressions across all 81 cells of that board.
- ✅ **FIXED** — Asymmetric cell margin ([cell_extraction.md](cell_extraction.md) Bug 3):
  `make_raw_cell` computes the margin once from the original span, restoring a symmetric crop.
- ✅ **FIXED** — "Solver doesn't understand unsolvable" ([solver_and_app.md](solver_and_app.md)):
  `solve_sudoku.board_has_conflicts()` checks the givens for row/col/box duplicates before
  solving; `main.py`/the frontend now report a distinct `"INVALID_BOARD"` result instead of
  lumping OCR misreads in with `"UNSOLVABLE"`.
- ✅ **FIXED** — App-layer concurrency/robustness gaps ([solver_and_app.md](solver_and_app.md)):
  each upload now gets a `uuid4` job id, its own upload filename and `results/<job_id>/`
  output directory (via a new optional `output_dir` arg to `main.py`); the polling timeout is
  no longer shared across both waits; a crashed subprocess is now detected and its stderr
  surfaced immediately instead of waiting out a generic 30s timeout.
- ⬜ **NOT FIXED (intentionally)** — Digit model architecture/checkpoint mismatch
  ([digit_recognition.md](digit_recognition.md)): `src/train.py` trains `SudokuDigitCNN` and
  saves a `state_dict` checkpoint, but the *active* `src/digit_recognizer.py` loads a whole
  pickled `LeNet` object from `models/mnist_lenet.pt` instead — an unrelated lineage; `train.py`'s
  own output models (`new_ds*.pt`, `old_black_white_ds*.pt`) are currently dead/unused at
  inference time. Left untouched because resolving it correctly requires running/comparing
  models, which was explicitly out of scope for the 2026-07-08 pass.
