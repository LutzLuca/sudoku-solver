def solve(sudoku) -> bool:
    idx = next_open_pos(sudoku)

    if idx is None:
        return True

    for num in range(1, 10):
        if is_safe(sudoku, idx, num):
            sudoku[idx] = num

            if solve(sudoku):
                return True
            sudoku[idx] = 0

    return False


def is_safe(sudoku, idx: int, val: int) -> bool:
    def block_coord(row: int, col: int):
        block_col, block_row = col - col % 3, row - row % 3

        for r in range(block_row, block_row + 3):
            for c in range(block_col, block_col + 3):
                yield (r * 9 + c)

    row, col = idx // 9, idx % 9

    return all(
        sudoku[row * 9 + v] != val and sudoku[v * 9 + col] != val for v in range(9)
    ) and all(sudoku[idx] != val for idx in block_coord(row, col))


def next_open_pos(sudoku) -> int | None:
    return next(
        (idx for idx, val in enumerate(sudoku) if val == 0),
        None,
    )
