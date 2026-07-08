# `src/solve_sudoku.py`, `app.py`, `main.py` — Reference

## Fix log (2026-07-08)

- **OCR-misread vs. genuinely unsolvable**: added `board_has_conflicts(board)` to
  `src/solve_sudoku.py` — checks the givens for a row/column/box duplicate *before* running
  `solve()`. `main.py` now calls it first and writes a new, distinct JSON value,
  `"INVALID_BOARD"`, instead of `"UNSOLVABLE"` when the conflict is in the givens themselves.
  `app/static/script.js` branches on this new value with its own message ("likely a misread
  digit, try a clearer photo"). Verified with direct unit-style checks (valid board, and boards
  with an injected row/column/box conflict), no model involved.
- **Shared-timer bug**: `app.py` previously reused one `start` timestamp across both polling
  loops, so the combined budget for both JSON files was 30s total, not 30s per file. Replaced
  with a `_wait_for_file()` helper that starts its own timer per call.
- **Concurrency/race conditions**: `app.py` now generates a `uuid4` job id per upload, used for
  both the upload filename prefix and a per-job `results/<job_id>/` output directory (passed
  to `main.py` as a new optional second CLI argument, `python main.py <image> [output_dir]`,
  defaulting to `results` when omitted so the CLI usage in the README/CLAUDE.md is unaffected).
  Concurrent uploads no longer share a filename or output path.
- **Silent 30s timeout on crash**: `_wait_for_file()` now checks `process.poll()` on every
  iteration; if `main.py` has already exited without producing the expected file, it reports
  the subprocess's actual stderr/exit code immediately instead of waiting out the rest of the
  timeout with a generic "Timeout" message. Verified with dummy subprocesses (crash-immediately,
  succeed-after-delay, hang-past-timeout) standing in for `main.py` — no model invoked.
- **Not changed**: the `except GridNotFoundError` branch in `main.py` is still present. It's
  confirmed unreachable in normal operation (see below) but harmless as a defensive fallback,
  so it was left in place rather than removed.

## Purpose / architecture

Three pieces, connected only through JSON files on disk (per the README:
"the app depends only on the JSON files, no other coupling"):

```
Browser (app/templates/index.html + app/static/script.js)
      │  POST /upload (multipart image)
      ▼
app.py (Flask)
  - saves upload to app/uploads/<secure_filename>
  - deletes results/00_original.json + results/00_solved.json if present
  - subprocess.Popen(["python", "main.py", filepath])   <- fire-and-forget
  - polls filesystem for results/00_original.json (<=30s)
  - polls filesystem for results/00_solved.json (<=30s)
  - returns {image, original, solved} as JSON
      ▲
      │  writes JSON files
main.py (CLI orchestrator, invoked as a fresh `python main.py <image>` process)
  1. prepare_output_dir("results")  -> shutil.rmtree + recreate
  2. preprocess()          (src/preprocess.py)
  3. extract()             (src/grid_extraction.py) — wrapped in
     try/except GridNotFoundError
  4. save_cells()           (src/cell_extraction.py)
  5. DigitRecognizer().predict_board(cells)  -> 9x9 int board (OCR, plain
     argmax over LeNet logits, no confidence check)
  6. print_board(board, ..., "results/00_original.json")   <- writes original
  7. solve(solved) from src/solve_sudoku.py (plain backtracking)
     - success -> print_board(solved, ..., "results/00_solved.json")
     - failure -> writes the literal JSON string "UNSOLVABLE" to
       results/00_solved.json
```

`src/solve_sudoku.py` itself is a minimal, textbook exhaustive backtracking
solver:

- `find_empty(board)` — first 0 cell in row-major order.
- `valid(board, row, col, num)` — checks num not already in row / column /
  3x3 box.
- `solve(board)` — recursive brute force: try 1-9 in the first empty cell,
  recurse, undo on failure. No constraint propagation (no naked-singles /
  hidden-singles pruning), no MRV (minimum-remaining-values) cell ordering,
  no memoization. It is exhaustive and *will* terminate with `False` for any
  board that is truly unsatisfiable — see "Known Issues" below for why the
  README's bug report is not about this exhaustiveness.

The frontend (`app/templates/index.html`, `app/static/script.js`,
`app/static/style.css`) is a static single page: pick a file, POST it to
`/upload`, then render two 9x9 `<table>` grids from the JSON response.
`script.js` does branch on the shape of `data.solved`:

```js
if (data.solved === "UNSOLVABLE") {
    // renders the red "This Sudoku is unsolvable" banner
} else {
    drawBoard(data.solved, "solvedBoard", true);   // expects a 2D array
}
```

So the two JSON contract shapes (`"UNSOLVABLE"` string vs 2D array) are each
handled explicitly — this part of the contract is honored correctly on the
frontend.

## Known Issues

### 1. "Solver sometimes doesn't understand a board is unsolvable" — root cause is upstream OCR, not the backtracking algorithm

The README (translated): *"the sudoku-solving model sometimes doesn't
understand [a board] is unsolvable — fix it."* Reading `solve()`/`valid()`
line by line: this is a complete, exhaustive backtracking search over a
static 9x9 board. For a board that is genuinely contradictory (duplicate
given digits already violating a row/column/box constraint, or a
well-formed-but-unsatisfiable puzzle), `find_empty` + `valid` + the
recursive retry/undo loop *will* eventually exhaust all 9^k branches and
`solve()` *will* return `False`. There is no code path where the algorithm
itself "thinks" an unsolvable board is solvable, nor a path where it loops
forever on an unsolvable board — it is a correct, terminating algorithm for
the stated problem (verified by inspection: no early-exit assumes success,
no memo/cache could return a stale `True`).

The real bug is one level up, in `main.py`'s `predict_board()` call
(`src/digit_recognizer.py`, `predict_board`, lines 113-130): each cell is
classified independently with a plain `torch.argmax(logits, dim=1)` and
*no* confidence thresholding, no cross-checking against sudoku constraints,
and no validation of the resulting board before it is handed to `solve()`.
If the CNN misreads even one digit (e.g. classifies a `6` as an `8` in a
cell that already shares a row/box with a real `8`), the resulting board now
contains a duplicate given — a state that is *identically* structured to a
genuinely-unsolvable board from `solve()`'s point of view. `solve()` will
correctly, exhaustively conclude `False`, `main.py` writes the literal
string `"UNSOLVABLE"` to `results/00_solved.json`, and the user sees "This
Sudoku is unsolvable" (`app/static/script.js` line 33) for what was actually
a solvable board with one misread digit. This matches the README complaint
much better than an algorithmic defect: the solver "doesn't understand" the
board is solvable because it was never given the real board.

**Evidence:**
- `src/solve_sudoku.py` has no duplicate-given check anywhere before the
  search starts — `valid()` is only ever called from inside `solve()` on
  cells `board[row][col] == 0`, so a duplicate that's already present among
  the *given* (non-zero) clues is never itself flagged; it just silently
  makes every candidate for the conflicting row/col/box invalid, which
  looks to the algorithm exactly like a hard/unsolvable puzzle.
- `src/digit_recognizer.py` `predict_board` (lines 113-130): `pred =
  torch.argmax(logits, dim=1).item()` with no softmax-confidence check, no
  second-best-guess fallback, and no post-hoc sudoku-constraint validation.
- `main.py` lines 181-196: `solved = [row[:] for row in board]` is handed
  straight to `solve()` with zero pre-validation.

**Suggested fix:** before calling `solve()` in `main.py`, run a cheap
O(81) validation pass over the non-zero givens (row/col/box duplicate
check — essentially `valid()` applied to every already-filled cell against
its peers). If a duplicate is found among the *givens themselves*, this is
almost certainly an OCR misread, not a genuinely unsolvable puzzle — fail
fast with a distinct JSON payload (e.g. `{"error": "invalid_board", "reason":
"duplicate digit detected — check for a misread cell"}`) instead of running
a doomed backtracking search and reporting a bare `"UNSOLVABLE"` that
conflates "no solution exists" with "we probably misread a digit." This also
lets the frontend show a more actionable message than "This Sudoku is
unsolvable" (e.g. "we may have misread a digit — try retaking the photo").

### 2. Naive backtracking has no worst-case time bound — can interact badly with `app.py`'s 30s polling timeout

`solve()` has no constraint propagation or MRV heuristic, so its worst case
is exponential. Because `find_empty` always scans row-major from `(0,0)`,
a board whose only contradiction is deep in the grid (e.g. a conflict
first detectable in row 8 or the bottom-right box) forces the search to
explore a large fraction of the tree for rows 0-7 before ever reaching the
conflict — the classic pathological case for plain backtracking sudoku
solvers (well documented behavior for this style of implementation, not
specific to this codebase). A CNN misread that introduces a contradiction
late in the grid (rather than an early cell, which fails fast) is
therefore plausibly *both* the source of a false "unsolvable" verdict *and*
a multi-second-to-multi-minute stall, depending on how empty/contradictory
the board ends up being.

Meanwhile `app.py` polls for `results/00_original.json` for up to 30s, then
separately for `results/00_solved.json` for up to another 30s (lines 55-74)
— but both timers actually share one `start = time.time()` taken before the
first poll loop (line 56) that is *never reset* before the second loop
(line 69 reuses the same `start`). That means the real end-to-end budget
for `main.py` to write *both* files is 30s total, not 60s as a naive reading
suggests — the second `while` loop's `time.time() - start > timeout` check
will already be close to (or past) 30 when it begins if the OCR/detection
stage was slow, immediately timing out the "solved" wait. A slow
`solve()` call — especially combined with the several-second Python/Torch/
OpenCV cold-start cost of spinning up a brand new subprocess for every
single request — can easily exceed this budget on a modest machine, so the
user gets a generic `{"error": "Timeout waiting for solved board"}` (line
72) even though `main.py` may still be running to completion, will finish
seconds/minutes later, and will still write `results/00_solved.json` —
except by then app.py has already returned an HTTP 500 and moved on, so the
result is silently discarded (and can also race with the *next* upload's
`prepare_output_dir` `rmtree`, see Issue 3).

**Suggested fix:** add the givens-validation fast-fail from Issue 1 (avoids
the slow path entirely for the common OCR-misread case); consider adding
MRV cell ordering to `find_empty` (pick the empty cell with fewest legal
candidates first) to bound worst-case runtime; fix the shared `start`
timer bug so each phase gets its own full budget, or better, replace
polling with a real completion signal (return code from the subprocess, or
call the pipeline in-process — see "Specs for future work").

### 3. Concurrency: fixed global JSON/output paths — concurrent requests corrupt each other

`ORIGINAL_JSON = "results/00_original.json"` and `SOLVED_JSON =
"results/00_solved.json"` (app.py lines 17-18) are process-wide constants,
not per-request. `main.py`'s `OUTPUT_DIR = "results"` is likewise a single
shared directory, and `prepare_output_dir()` (main.py lines 125-129) does
`shutil.rmtree(output_dir)` **unconditionally at the start of every run**.
If two browser tabs/users upload at (or near) the same time:

- Request B's `main.py` subprocess can `rmtree("results")` out from under
  request A's subprocess while A is still writing (or about to write)
  `results/00_original.json` / `results/00_solved.json` — a `FileNotFoundError`
  or silently-lost write is possible, or A's poll loop in `app.py` starts
  reading B's half-written / wrong-board JSON.
- Both requests' `app.py` handlers poll the *same* two file paths, so
  whichever subprocess writes first "wins" and both HTTP responses can end
  up returning the same (possibly mismatched image vs solution) board.
- `app.py`'s upfront cleanup (`for f in [ORIGINAL_JSON, SOLVED_JSON]: os.remove(f)`,
  lines 45-47) run by request B can delete request A's freshly-written
  files while A's poll loop is between checks, causing A's loop to keep
  waiting (or, worse, A already read stale/partial content).

There is no locking, no per-session/request ID, and no isolation — this is
a straightforward, easily-reproducible race with 2+ concurrent users.

**Suggested fix:** derive a unique per-request ID (e.g. `uuid4()` or the
Flask request context) and use it to namespace both the upload path and the
`results/<id>/` output directory / JSON filenames; pass that directory into
`main.py` (already parameterized via `OUTPUT_DIR`/`prepare_output_dir`) so
concurrent runs never share files. Longer term, drop the file-polling
handoff entirely (see below).

### 4. `secure_filename` + shared `UPLOAD_FOLDER` — same-name uploads silently overwrite

`filename = secure_filename(file.filename); filepath = os.path.join(UPLOAD_FOLDER, filename)`
(app.py lines 39-40) with a single shared `app/uploads/` directory means two
different uploads named e.g. `img.jpg` (from the same or different users)
collide: the second `file.save(filepath)` silently overwrites the first
with no warning, no uniqueness check, and no cleanup policy (uploads
accumulate forever otherwise). Combined with Issue 3's shared `results/`
directory, this compounds the concurrent-request corruption risk.
**Suggested fix:** same per-request UUID namespacing as Issue 3 applied to
the upload path.

### 5. `app.py` never inspects the subprocess's exit code or stderr

`process = subprocess.Popen(["python", "main.py", filepath])` (app.py lines
50-52) captures the `Popen` handle only to `.kill()` it on timeout — its
return code and stdout/stderr are never checked. If `main.py` crashes
immediately (bad image, missing model file, unhandled exception anywhere
in `preprocess`/`extract`/`save_cells`/`DigitRecognizer`), `app.py` has no
way to know *why* — it just keeps polling until the 30s budget elapses and
returns a generic `"Timeout waiting for original board"`, which is
misleading (it wasn't a slow computation, the process already died).
**Suggested fix:** after `Popen`, either poll `process.poll()` alongside the
file checks and break early with the real stderr (`stdout=subprocess.PIPE,
stderr=subprocess.PIPE` at construction) surfaced in the JSON error
response, or (preferred) call the pipeline as an in-process function so
exceptions propagate normally through Flask's error handling.

### 6. `except GridNotFoundError` in `main.py` is dead code

`main.py` (lines 94, 162-166) still does:

```python
from src.grid_extraction import extract, GridNotFoundError
...
try:
    corners, warped, _ = extract(pre, image, OUTPUT_DIR)
except GridNotFoundError:
    print("Grid not found")
    return
```

But `src/grid_extraction.py`'s `GridNotFoundError` class docstring states
explicitly: *"Kept for backwards compatibility with older callers. extract()
no longer raises this in normal operation... but the class stays importable
in case something else still catches it."* A `Grep` for `raise` across
`src/grid_extraction.py` confirms there is no `raise GridNotFoundError(...)`
anywhere in the file — `extract()`'s 3rd stage ("Last Resort / full-image
fallback," see `docs/grid_extraction.md`) always succeeds by treating the
whole image as the grid instead of raising. So this `except` branch is
currently unreachable dead code; `main.py` cannot fail this way with the
current `grid_extraction.py`, and the fallback-to-full-image behavior means
a badly-cropped/rotated photo silently proceeds with a nonsense warp rather
than surfacing a clear "grid not found" error to the user — arguably a
regression in error-reporting relative to what the still-present except
branch implies. **Suggested fix:** either remove the dead `except` branch
(and the now-vestigial import) to avoid misleading future readers, or
(better) have the "Last Resort" stage in `extract()` optionally re-raise
`GridNotFoundError` when the full-image fallback also fails some sanity
check, so `main.py`'s existing handling becomes meaningful again.

## Specs for future work

**JSON contract** (the only coupling between `main.py`/`app.py` per the
README):
- `results/00_original.json` — always a 9x9 array of ints 0-9 (0 = blank),
  written unconditionally by `main.py` right after OCR, before solving.
- `results/00_solved.json` — one of exactly two shapes:
  - a 9x9 array of ints 1-9 (the completed solution), or
  - the literal JSON string `"UNSOLVABLE"` (note: not an object/error code,
    just a bare string — `app/static/script.js` line 29 checks
    `data.solved === "UNSOLVABLE"` by strict equality).
- Any future producer of these files (a different solver, a rewritten
  `main.py`, a notebook, etc.) only needs to emit these two files in this
  shape at these paths for the existing Flask app + frontend to keep
  working unmodified — this is the contract the README refers to.

**CLI usage:** `python main.py <image_path>` — exactly one positional arg
(image file path); `main.py` exits early with a usage message if
`len(sys.argv) != 2`, and `sys.exit("Cannot read image")` if
`cv2.imread` returns `None`.

**Current app.py invocation model and what would need to change:**
- Today: `app.py` shells out via `subprocess.Popen(["python", "main.py",
  filepath])` and hands off entirely through the two shared JSON file
  paths, discovered by polling `os.path.exists` in a loop with a combined
  30s budget (see Issue 2 for the shared-timer bug) and no exit-code/stderr
  visibility (Issue 5).
- To make this robust: refactor `main.py`'s `main()` body into an
  importable function (e.g. `run_pipeline(image_path, output_dir) -> (board,
  solved_or_none)`) that `app.py` calls directly in-process (or via a task
  queue/worker for real concurrency control), returning the board data
  directly rather than round-tripping through the filesystem. This removes
  the polling loop, the shared-path race (Issue 3), and the swallowed
  exceptions (Issue 5) in one change, and makes the per-request output
  namespacing in Issue 3's fix straightforward (pass a request-scoped
  `output_dir`/id into the function call instead of relying on a global
  constant).
