# Digit Recognition Subsystem

This document covers the part of the pipeline responsible for turning a
warped, per-cell Sudoku image into a digit (0 = empty, 1-9 = recognized
digit): `src/digit_model.py`, `src/digit_recognizer.py`, and `src/train.py`,
plus how the loose scripts `main.py` and `test.py` relate to them.

## Fix log (2026-07-08)

**Intentionally not fixed.** The 2026-07-08 bug-fix pass covered every other section
(`preprocess.py`, `grid_extraction.py`, `cell_extraction.py`, `solve_sudoku.py`, `app.py`,
`main.py`) but explicitly left this file's model/checkpoint mismatch untouched, per direction
not to run or load any model as part of that work. Deciding which architecture+checkpoint
pairing is actually correct, and confirming a fix doesn't regress accuracy, requires running
inference — so the mismatch documented below is still live. Treat this doc's "Known Issues"
section as an accurate, still-open punch list for a future pass that is allowed to run models.

## Purpose of each file

### `src/digit_model.py`
Defines `SudokuDigitCNN`, a small custom CNN:
- `features`: Conv2d(1→32) + BN + ReLU + MaxPool → Conv2d(32→64) + BN + ReLU
  + MaxPool → Conv2d(64→128) + BN + ReLU → AdaptiveAvgPool2d((4,4))
- `classifier`: Flatten → Dropout(0.25) → Linear(128*4*4→128) → ReLU →
  Dropout(0.15) → Linear(128→10)
- Input: single-channel (grayscale) image of any size (adaptive pool
  normalizes spatial size to 4x4 before the classifier), output: 10 logits.

This is the "repo-native" architecture — the one this project's own
training code is built around.

### `src/train.py`
Standalone training entrypoint, run as `python -m src.train`. It:
- Loads a folder-structured dataset (`ImageFolder`) of pre-cropped digit
  images from `data/processed/digits/{train,val}`.
- Applies grayscale + resize(28,28) + augmentation (rotation, affine,
  color jitter) for train, and grayscale + resize(28,28) only for val —
  **`ToTensor()` only, no `Normalize` call**, so tensors stay in `[0,1]`.
- Trains `SudokuDigitCNN` (from `digit_model.py`) with Adam + CrossEntropy.
- Saves the **best** validation-accuracy checkpoint as a dict:
  `{"model_state_dict": model.state_dict(), "class_to_idx": ...}` — i.e. a
  `state_dict` checkpoint, not a pickled model object. It must be loaded
  with `SudokuDigitCNN(...).load_state_dict(checkpoint["model_state_dict"])`.
- README documents actual invocations such as
  `python -m src.train --data-dir data/processed/digits --epochs 10 --output models/new_ds_10_epoch.pt`.

### `src/digit_recognizer.py`
Defines `DigitRecognizer`, which is supposed to load a trained model and
expose `predict_board(cells)` for the main pipeline. The file currently
contains **two full implementations**:

1. A large commented-out block at the top (the "OLD" version): imports
   `SudokuDigitCNN` from `digit_model.py`, points `MODEL_PATH` at
   `models/old_black_white_ds.pt`, and loads it correctly via
   `checkpoint = torch.load(...)` then
   `model.load_state_dict(checkpoint["model_state_dict"])` — this is the
   loading style that actually matches what `train.py` produces.
2. The active "NEW" version below it: re-defines an inline `LeNet` class
   (conv1 1→6 k5 pad2, conv2 6→16 k5, fc1 16*5*5→120, fc2 120→84,
   fc3 84→10 — the classic LeNet-5 shape), sets
   `MODEL_PATH = Path("models/mnist_lenet.pt")`, and loads it with
   `self.model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)`
   — i.e. it expects the **entire model object** to have been pickled
   (`torch.save(model, ...)`, not `torch.save(model.state_dict(), ...)`),
   then calls `.eval()` directly on the unpickled object.

`predict_board(cells)` (both versions) iterates `cells`, skips
`cell.is_empty`, resizes `cell.image` to 28x28, applies the module-level
`transform`, runs the model, argmaxes, and reshapes the flat 81-length
list into a 9x9 board.

## Architecture summary

| | `SudokuDigitCNN` (digit_model.py) | inline `LeNet` (digit_recognizer.py / main.py) |
|---|---|---|
| Conv layers | 3x Conv+BN+ReLU+Pool(ish), channels 1→32→64→128 | 2x Conv+ReLU+Pool, channels 1→6→16, kernel 5 |
| Pooling before FC | AdaptiveAvgPool2d(4,4) (size-agnostic) | fixed-shape flatten `16*5*5` (assumes exactly 28x28 input with padding=2 on conv1) |
| FC head | 128*4*4→128→10, with dropout | 16*5*5→120→84→10, no dropout |
| Saved as | `state_dict` + `class_to_idx` dict (`torch.save({...}, path)`) | whole pickled `nn.Module` object (`torch.save(model, path)`), loaded with `weights_only=False` |
| Produced by | `src/train.py` | some external/unknown training script (not present in this repo) — `mnist_lenet.pt` looks like it was trained directly on MNIST, not on this repo's Sudoku cell dataset |

## What is actually used at inference time today

**`models/mnist_lenet.pt`**, loaded by the *active* (uncommented) code
in `src/digit_recognizer.py`, via the inline `LeNet` architecture and
`torch.load(..., weights_only=False)`.

Directory listing of `models/` (sizes only, files not deserialized):

```
mnist_lenet.pt                  183,088 bytes   (2026-07-06 10:42)
new_ds.pt                     1,435,833 bytes   (2026-07-06 02:13)
new_ds_10_epoch.pt            1,437,456 bytes   (2026-07-06 03:53)
old_black_white_ds.pt         1,437,334 bytes   (2026-07-05 15:04)
old_black_white_ds_10_epochs.pt 1,437,334 bytes (2026-07-07 12:24)
```

`mnist_lenet.pt` is an order of magnitude smaller than the other four,
consistent with LeNet's much smaller parameter count vs.
`SudokuDigitCNN`'s 128-channel conv layers and 128*4*4→128 linear layer.
The other four files' size and naming (`old_black_white_ds*`,
`new_ds*`) line up with checkpoints produced by `src/train.py` for
`SudokuDigitCNN` (the OLD code in `digit_recognizer.py` explicitly names
`models/old_black_white_ds.pt` as its `MODEL_PATH`; README shows
`train.py` writing `models/new_ds_10_epoch.pt`). None of these four are
referenced by any currently-active loading code path — the OLD
`digit_recognizer.py` code that would load them is commented out.

## Known Issues

### 1. Architecture/checkpoint mismatch (primary bug)
- **Evidence**: `train.py` line 131:
  `torch.save({"model_state_dict": model.state_dict(), "class_to_idx": ...}, output_path)`
  for a `SudokuDigitCNN` instance. The only code that knows how to load
  that shape (`SudokuDigitCNN(...).load_state_dict(checkpoint["model_state_dict"])`)
  is the commented-out OLD block in `digit_recognizer.py` (lines 23-56).
  The active code (lines 106-134) instead does
  `torch.load(MODEL_PATH, map_location="cpu", weights_only=False)` and
  calls `.eval()` directly on the result, expecting `MODEL_PATH` to be a
  whole pickled `LeNet` object — and points it at `models/mnist_lenet.pt`,
  a file with no known relationship to `train.py` or `SudokuDigitCNN`.
- **Consequence (a)**: every model this repo's own `train.py` produces
  (`new_ds.pt`, `new_ds_10_epoch.pt`, `old_black_white_ds.pt`,
  `old_black_white_ds_10_epochs.pt`) is currently dead weight — none of
  them is loaded at inference time. Training a new model with
  `python -m src.train` has **no effect** on what `main.py` actually
  predicts, unless a developer also hand-edits `digit_recognizer.py` to
  restore the OLD code path.
- **Consequence (b)**: if someone naively points
  `MODEL_PATH = Path("models/new_ds.pt")` (or any of the other three)
  into the *active* loader, `torch.load(...)` returns a plain `dict`
  (containing `model_state_dict` and `class_to_idx` keys), not an
  `nn.Module`. The very next line, `self.model.eval()`, will raise
  `AttributeError: 'dict' object has no attribute 'eval'`. This is an
  easy trap for a future engineer who reads the file/directory names and
  assumes they're interchangeable with `mnist_lenet.pt`.
- **Recommended fix**: pick one checkpoint convention repo-wide (state_dict
  is the safer/standard choice — avoids `weights_only=False` pickle-trust
  issues too). Make `DigitRecognizer.__init__` always do
  `model = SudokuDigitCNN(); model.load_state_dict(torch.load(path)["model_state_dict"])`,
  point `MODEL_PATH` at whichever of `new_ds_10_epoch.pt` /
  `old_black_white_ds_10_epochs.pt` is the intended "best" model, and
  either delete `mnist_lenet.pt`/the inline `LeNet` path or clearly branch
  on an explicit `--arch` flag if both lineages must be kept.

### 2. Duplicated `LeNet` class definition
- **Evidence**: the identical `LeNet` class body (conv1/conv2/fc1/fc2/fc3,
  same layer sizes, same forward) is defined independently in both
  `src/digit_recognizer.py` (lines 70-88) and `main.py` (lines 102-120).
  `main.py` imports `DigitRecognizer` from `digit_recognizer.py` but does
  not import or reuse its `LeNet` — it maintains its own copy.
- **Risk**: no single source of truth. If one copy is edited (e.g. to
  change a layer width) and the other isn't, `torch.load(..., weights_only=False)`
  on a pickle that references `main.LeNet` vs. `src.digit_recognizer.LeNet`
  can unpickle to two different classes with different shapes, or fail
  outright if the pickled module path no longer matches the class actually
  imported at that name. It's a silent-drift trap, not a currently-thrown
  error, since nothing loads `main.LeNet` today (it's dead code in `main.py`
  as far as the actual recognizer path is concerned).
- **Recommended fix**: define `LeNet` in exactly one module (e.g.
  `src/digit_model.py`, alongside `SudokuDigitCNN`) and import it wherever
  needed. Delete the copy in `main.py`.

### 3. Normalization mismatch between the two pipelines
- **Evidence**: `digit_recognizer.py`'s active `transform` (lines 96-102)
  ends with `transforms.Normalize((0.1307,), (0.3081,))` (canonical MNIST
  mean/std). `train.py`'s `train_tfms`/`val_tfms` (lines 30-46) end with
  `transforms.ToTensor()` only — no `Normalize` call, so pixel values fed
  to `SudokuDigitCNN` stay in raw `[0,1]` range.
- **Confirmed real, but not itself "the bug"**: taken in isolation, each
  pipeline is internally consistent — *if* `mnist_lenet.pt` was trained on
  real MNIST data using the same `(0.1307, 0.3081)` normalization (plausible,
  since those are the standard MNIST constants and `test.py` at the repo
  root uses the identical normalization for what its comments describe as
  MNIST-style preprocessing), then that pairing is fine for *that* model.
  The real issue is that this reveals `SudokuDigitCNN`+`train.py`
  (unnormalized, trained on this repo's own cropped/binarized Sudoku
  digit images) and `LeNet`+`mnist_lenet.pt` (normalized, apparently
  trained on stock MNIST) are two **entirely separate, non-interchangeable
  pipelines** that happen to coexist in the same files. Swapping the
  checkpoint without also swapping the transform (or vice versa) silently
  produces garbage predictions rather than a crash, which is worse than
  issue #1's crash-on-load failure mode.
- **Recommended fix**: document (or better, encode in code, e.g. as a
  class attribute on each model wrapper) which transform pipeline goes
  with which architecture/checkpoint, so the pairing can't be
  accidentally split.

### 4. `test.py` is orphaned scratch code
- **Evidence**: `test.py` at the repo root defines a *third* architecture
  (`ResNet18MNIST`, wrapping `torchvision.models.resnet18`) and a *fourth*
  LeNet variant (`LeNet5`, with `AvgPool2d` instead of `MaxPool2d` and a
  `16*4*4` flatten instead of `16*5*5` — i.e. not shape-compatible with
  either of the other two LeNet-ish definitions). It hardcodes a
  developer-specific absolute Windows path
  (`C:\Users\T.Abdellahi\Desktop\...`) as the image to run inference on,
  and its `load_model()` tries `torch.load('mnist_lenet5.pth', ...)` — a
  filename that does not exist anywhere under `models/` (the closest match,
  `mnist_lenet.pt`, has a different name and extension). The `try/except`
  around that load silently falls back to random/untrained weights if the
  file is missing, so running this script "successfully" gives no signal
  about real model quality.
- **Conclusion**: this is dead/orphaned experimentation code, unrelated
  to the maintained `main.py` → `DigitRecognizer` pipeline. It should not
  be treated as a reference implementation for anything.
- **Recommended fix**: delete `test.py`, or move it to a clearly-named
  `scratch/` or `experiments/` folder with a comment marking it as
  non-authoritative, so it doesn't get mistaken for part of the pipeline
  by a future contributor.

### 5. README's own model-swap note matches these findings
`README.md` (around line 285) says, in Persian: *"مدل رو اگه خواستی عوض
کنی باید اینا عوض:"* ("if you want to change the model, you need to
change these:") followed by *"مین"* (main), *"دیجیت مدل"* (digit_model),
*"دیجیت ریکگنایزر"* (digit_recognizer), *"ترین"* (train). This matches
exactly what was found: those are precisely the files with duplicated
architecture definitions (`LeNet` in both `main.py` and
`digit_recognizer.py`) and hardcoded, mismatched paths/architectures
(`digit_recognizer.py`'s `MODEL_PATH`, `train.py`'s `--output` default).
The fact that swapping models requires touching four separate files by
hand, with no single loading abstraction, is itself the design smell —
there is no `ModelRegistry`/factory that pairs an architecture, a
checkpoint format, and a preprocessing transform as one unit.

## Specs for future work

Before touching digit recognition, a future engineer must know:

1. **Which model is live today**: `models/mnist_lenet.pt`, loaded as a
   whole pickled object by the active (non-commented) code in
   `src/digit_recognizer.py`, using the inline `LeNet` architecture also
   duplicated in `main.py`. `SudokuDigitCNN` and all of
   `new_ds.pt` / `new_ds_10_epoch.pt` / `old_black_white_ds.pt` /
   `old_black_white_ds_10_epochs.pt` are currently unused at inference
   time, even though `train.py` (this repo's own training code) only
   knows how to produce checkpoints in that state-dict format.

2. **What `predict_board(cells)` expects `cell.image` to look like**:
   per `src/cell_extraction.py`, non-empty cells have already been run
   through `process_cell` → `clean_noise` by the time `DigitRecognizer`
   sees them. That means `cell.image` is:
   - a single-channel `uint8` array,
   - already **binary** (values only `0` or `255`, from
     `cv2.threshold(..., THRESH_BINARY | THRESH_OTSU)` followed by
     connected-components filtering in `clean_noise`),
   - background = `0` (black), digit strokes = `255` (white) — the
     `WHITE_RATIO_FOR_INVERSE` check in `process_cell` auto-inverts so
     white-ratio stays low (digit-as-foreground, not background-as-white),
   - with grid border lines near cell edges removed (Hough-line based),
     and small noise components below `MIN_NOISE_AREA` (15 px) removed,
   - **not yet resized** — `predict_board` itself calls
     `cv2.resize(cell.image, (28, 28))` before handing off to the
     model's `transform`.
   Any new model/architecture must either consume this same binary
   0/255 format or the preprocessing must be adapted accordingly — do not
   assume grayscale/antialiased input like raw MNIST digits.

3. **What changing the model requires touching**, per the README's own
   note and confirmed above: `main.py` (duplicate `LeNet` def +
   `DigitRecognizer` instantiation), `src/digit_model.py`
   (`SudokuDigitCNN` def, if that lineage is kept), `src/digit_recognizer.py`
   (`MODEL_PATH`, architecture import/class, `transform`), and
   `src/train.py` (architecture import, output checkpoint format). There is
   no single abstraction that owns "architecture + checkpoint path +
   preprocessing transform" as one unit, so all four must be kept in sync
   by hand today.

4. **Do not treat `test.py` as reference code** — it is orphaned, has a
   hardcoded personal path, references a nonexistent `mnist_lenet5.pth`,
   and defines two more architectures (`ResNet18MNIST`, a differently-shaped
   `LeNet5`) not used anywhere else in the pipeline.

5. **Before "just changing the checkpoint path"**: confirm the
   preprocessing `transform` (Normalize or not) and architecture class
   match the checkpoint's training pipeline — issue #3 above shows this
   can fail silently (wrong predictions, no crash) rather than loudly.
