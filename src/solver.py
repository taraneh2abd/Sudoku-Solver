from __future__ import annotations


Grid = list[list[int]]


def validate_grid(grid: Grid) -> bool:
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        return False
    return all(_valid_group(group) for group in _all_groups(grid))


def _valid_group(values: list[int]) -> bool:
    seen: set[int] = set()
    for value in values:
        if value == 0:
            continue
        if value < 1 or value > 9 or value in seen:
            return False
        seen.add(value)
    return True


def _all_groups(grid: Grid) -> list[list[int]]:
    groups: list[list[int]] = []
    groups.extend([row[:] for row in grid])
    groups.extend([[grid[row][col] for row in range(9)] for col in range(9)])
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            groups.append(
                [
                    grid[row][col]
                    for row in range(box_row, box_row + 3)
                    for col in range(box_col, box_col + 3)
                ]
            )
    return groups


def find_empty(grid: Grid) -> tuple[int, int] | None:
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                return row, col
    return None


def is_valid_move(grid: Grid, row: int, col: int, value: int) -> bool:
    if any(grid[row][c] == value for c in range(9)):
        return False
    if any(grid[r][col] == value for r in range(9)):
        return False

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] == value:
                return False
    return True


def solve_sudoku(grid: Grid) -> bool:
    empty = find_empty(grid)
    if empty is None:
        return validate_grid(grid)

    row, col = empty
    for value in range(1, 10):
        if is_valid_move(grid, row, col, value):
            grid[row][col] = value
            if solve_sudoku(grid):
                return True
            grid[row][col] = 0
    return False
