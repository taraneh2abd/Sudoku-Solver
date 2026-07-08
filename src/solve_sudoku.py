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