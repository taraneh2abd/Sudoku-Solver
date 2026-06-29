# Sudoku Solver From Image

This project detects a Sudoku board from an input image, extracts its 81 cells,
recognizes digits with a CNN, solves the puzzle with backtracking, and overlays
the final answer on the original image.

## Features

- OpenCV image-processing pipeline
- Perspective correction and 81-cell extraction
- Empty-cell detection
- CNN digit classifier scaffold with training and evaluation scripts
- Backtracking Sudoku solver
- Optional answer overlay on the original perspective
- Debug outputs for report writing and failure analysis

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run On An Image

```bash
python main.py --image "C:\Users\T.Abdellahi\Desktop\term8\vision\proj\Sudoku-Solver\data\raw\lexski\train\00000.jpg" --model models/digit_cnn.pt --debug
```

Outputs are written to `outputs/`.

## Train Digit Model

Recommended dataset for this project:

- HuggingFace: `Lexski/sudoku-image-recognition`

Download and prepare it with:

```bash
python -m src.prepare_dataset
```

This creates:

- Raw Sudoku images in `data/raw/lexski/`
- CNN cell crops in `data/processed/digits/`
- Labels/keypoints metadata in `data/processed/lexski_metadata/`

The digit dataset is prepared in this format:

```text
data/processed/digits/
  train/
    0/
    1/
    ...
    9/
  val/
    0/
    1/
    ...
    9/
```

Class `0` means empty cell. Classes `1..9` are Sudoku digits.

Then run:

```bash
python -m src.train --data-dir data/processed/digits --epochs 15 --output models/digit_cnn.pt
```

For a quick download/debug run:

```bash
python -m src.prepare_dataset --max-per-split 20
```

## Important Files

- `main.py`: end-to-end CLI
- `src/prepare_dataset.py`: downloads and prepares the selected Sudoku dataset
- `src/image_pipeline.py`: board detection and perspective transform
- `src/cell_extraction.py`: cell extraction and empty-cell detection
- `src/digit_model.py`: CNN architecture
- `src/train.py`: training and evaluation script
- `src/predict.py`: model loading and cell prediction
- `src/solver.py`: Sudoku validation and backtracking solver
- `src/overlay.py`: draw solved digits on original image
- `reports/final_report.md`: report template

## Notes

The model weights are not included by default. Train the CNN on MNIST/Hoda,
synthetic cells, or manually collected Sudoku cell crops, then place the final
weights at `models/digit_cnn.pt`.
