from solver import solve
import numpy as np


def is_valid_sudoku(sudoku) -> bool:
    def is_valid_unit(unit) -> bool:
        return all(num in unit for num in range(1, 10))

    def get_blocks(sudoku):
        for block_y in range(0, 9, 3):
            for block_x in range(0, 9, 3):
                yield sudoku[block_y : block_y + 3, block_x : block_x + 3].flatten()

    sudoku = np.array(sudoku).reshape(9, 9)

    return all(
        (is_valid_unit(block) for block in get_blocks(sudoku))
        and (
            is_valid_unit(sudoku[v : v + 1, :].flatten())
            and is_valid_unit(sudoku[:, v : v + 1].flatten())
            for v in range(0, 9)
        )
    )


# fmt: off
test_cases = [
    [
        5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9,
    ],
    [
        5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9,
    ],
    [
        0, 0, 3, 0, 2, 0, 6, 0, 0,
        9, 0, 0, 3, 0, 5, 0, 0, 1,
        0, 0, 1, 8, 0, 6, 4, 0, 0,
        0, 0, 8, 1, 0, 2, 9, 0, 0,
        7, 0, 0, 0, 0, 0, 0, 0, 8,
        0, 0, 6, 7, 0, 8, 2, 0, 0,
        0, 0, 2, 6, 0, 9, 5, 0, 0,
        8, 0, 0, 2, 0, 3, 0, 0, 9,
        0, 0, 5, 0, 1, 0, 3, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 1, 0,
        4, 0, 0, 0, 0, 5, 0, 0, 0,
        0, 2, 0, 7, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 6, 0, 0, 0, 3,
        0, 0, 0, 0, 0, 0, 5, 0, 0,
        0, 6, 0, 9, 0, 0, 0, 0, 0,
        0, 0, 4, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 7, 0, 0, 4,
        0, 0, 0, 0, 8, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 6, 0, 0, 0,
        0, 9, 0, 0, 0, 0, 0, 0, 0,
        3, 0, 0, 0, 0, 8, 0, 0, 0,
        0, 0, 0, 9, 0, 0, 7, 0, 0,
        0, 0, 0, 5, 0, 0, 0, 0, 4,
        0, 0, 0, 0, 8, 0, 0, 7, 0,
        0, 0, 0, 4, 0, 0, 5, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
]
# fmt: on

for sudoku in test_cases:
    assert solve(sudoku), "Sudoku should be solvable"
    assert is_valid_sudoku(sudoku), "Sudoku should be solved correctly"
