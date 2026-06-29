from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.solver import solve_sudoku, validate_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve a Sudoku grid stored as JSON.")
    parser.add_argument("--grid", required=True, help="Path to JSON file containing a 9x9 integer grid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    if not validate_grid(grid):
        raise ValueError("Input grid is invalid.")
    if not solve_sudoku(grid):
        raise ValueError("Sudoku has no solution.")
    print(json.dumps(grid, indent=2))


if __name__ == "__main__":
    main()
