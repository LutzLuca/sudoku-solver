def solve(sudoku) -> bool:
    next_pos = next_open_pos(sudoku)

    if next_pos is None:
        return True

    row, col = next_pos

    for num in range(1, 10):
        if is_safe(sudoku, next_pos, num):
            sudoku[row][col] = num

            if solve(sudoku):
                return True
            sudoku[row][col] = 0

    return False


def is_safe(sudoku, pos: tuple[int, int], val: int) -> bool:
    def block_coord(row: int, col: int):
        block_row, block_col = col - col % 3, row - row % 3

        for c in range(block_row, block_row + 3):
            for r in range(block_col, block_col + 3):
                yield (r, c)

    row, col = pos

    return all(
        sudoku[row][v] != val and sudoku[v][col] != val for v in range(9)
    ) and all(sudoku[r][c] != val for r, c in block_coord(row, col))


def next_open_pos(sudoku) -> tuple[int, int] | None:
    return next(
        (
            (r, c)
            for r, row in enumerate(sudoku)
            for c, val in enumerate(row)
            if val == 0
        ),
        None,
    )
