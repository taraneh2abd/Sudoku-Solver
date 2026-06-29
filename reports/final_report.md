# Sudoku Solver Final Report

## 1. Project Goal

The goal is to build a complete computer-vision system that detects a Sudoku
grid from a real image, extracts cells, recognizes digits, solves the puzzle,
and displays the final answer on the original image.

## 2. Image Processing Pipeline

1. Convert input image to grayscale.
2. Apply Gaussian blur to reduce noise.
3. Use adaptive thresholding to isolate dark grid lines and digits.
4. Detect contours and select the largest valid four-corner contour.
5. Apply perspective transform to create a square top-down board.
6. Split the board into 81 cells.
7. Remove cell borders and normalize candidate digit crops.
8. Detect empty cells using foreground-pixel ratio.

## 3. Digit Recognition

The classifier is a 10-class CNN:

- Class 0: empty cell
- Classes 1-9: Sudoku digits

The model uses convolution, batch normalization, max pooling, dropout, and a
fully connected classifier. Training uses augmentation, normalization, and
CrossEntropyLoss.

## 4. Sudoku Solver

The solver receives a 9x9 matrix. Empty cells are represented with zero. The
algorithm validates rows, columns, and 3x3 boxes, then solves the puzzle using
standard backtracking.

## 5. Failure Analysis

Common failure cases:

- Low contrast between board lines and background
- Strong perspective distortion
- Shadows across the grid
- Blurry digits
- Digits touching grid lines
- Very thin or broken grid lines

Recommended improvements:

- More real Sudoku images
- Synthetic augmentation with perspective distortion and shadows
- Better empty-cell classifier
- Fine-tuning on cropped cells from real photos

## 6. Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python -m src.train --data-dir data/processed/digits --epochs 15 --output models/digit_cnn.pt
```

Run final system:

```bash
python main.py --image data/raw/sample.jpg --model models/digit_cnn.pt --debug
```
