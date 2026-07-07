def find_empty(board):

    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c

    return None


def valid(board, row, col, num):

    if num in board[row]:
        return False

    for r in range(9):
        if board[r][col] == num:
            return False

    br = (row // 3) * 3
    bc = (col // 3) * 3

    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if board[r][c] == num:
                return False

    return True


def board_has_conflicts(board):
    """True if the given (non-zero) cells already break a row/column/box rule.

    solve() is an exhaustive backtracking search: it always terminates and
    correctly returns False for any board with no solution, whether that's
    because the givens are genuinely contradictory (e.g. two "5"s in the
    same row) or because the puzzle is merely hard. Callers that want to
    report "this looks like a misread digit" instead of a generic
    "unsolvable" need to check this *before* calling solve(), since solve()
    itself cannot distinguish the two cases - both just come back False.
    """
    for row in range(9):
        seen = set()
        for col in range(9):
            num = board[row][col]
            if num == 0:
                continue
            if num in seen:
                return True
            seen.add(num)

    for col in range(9):
        seen = set()
        for row in range(9):
            num = board[row][col]
            if num == 0:
                continue
            if num in seen:
                return True
            seen.add(num)

    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            seen = set()
            for row in range(br, br + 3):
                for col in range(bc, bc + 3):
                    num = board[row][col]
                    if num == 0:
                        continue
                    if num in seen:
                        return True
                    seen.add(num)

    return False


def solve(board):

    pos = find_empty(board)

    if pos is None:
        return True

    row, col = pos

    for num in range(1, 10):

        if valid(board, row, col, num):

            board[row][col] = num

            if solve(board):
                return True

            board[row][col] = 0

    return False