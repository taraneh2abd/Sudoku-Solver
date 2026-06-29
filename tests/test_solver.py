from src.solver import solve_sudoku, validate_grid


def test_solve_known_grid() -> None:
    grid = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    assert validate_grid(grid)
    assert solve_sudoku(grid)
    assert grid[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]


def test_reject_invalid_grid() -> None:
    grid = [[0 for _ in range(9)] for _ in range(9)]
    grid[0][0] = 5
    grid[0][1] = 5
    assert not validate_grid(grid)
