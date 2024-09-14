from solver import solve


def is_valid_sudoku(grid):
    def is_valid(unit):
        return set(unit) == set(range(1, 10))

    def get_subgrid(grid, row_start, col_start):
        return [grid[row_start + r][col_start + c] for r in range(3) for c in range(3)]

    for row in grid:
        if not is_valid(row):
            return False

    for col in range(9):
        if not is_valid([grid[row][col] for row in range(9)]):
            return False

    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            if not is_valid(get_subgrid(grid, r, c)):
                return False

    return True


test_cases = [
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ],
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ],
    [
        [0, 0, 3, 0, 2, 0, 6, 0, 0],
        [9, 0, 0, 3, 0, 5, 0, 0, 1],
        [0, 0, 1, 8, 0, 6, 4, 0, 0],
        [0, 0, 8, 1, 0, 2, 9, 0, 0],
        [7, 0, 0, 0, 0, 0, 0, 0, 8],
        [0, 0, 6, 7, 0, 8, 2, 0, 0],
        [0, 0, 2, 6, 0, 9, 5, 0, 0],
        [8, 0, 0, 2, 0, 3, 0, 0, 9],
        [0, 0, 5, 0, 1, 0, 3, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [4, 0, 0, 0, 0, 5, 0, 0, 0],
        [0, 2, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 6, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 5, 0, 0],
        [0, 6, 0, 9, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 7, 0, 0, 4],
        [0, 0, 0, 0, 8, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 6, 0, 0, 0],
        [0, 9, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 8, 0, 0, 0],
        [0, 0, 0, 9, 0, 0, 7, 0, 0],
        [0, 0, 0, 5, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 8, 0, 0, 7, 0],
        [0, 0, 0, 4, 0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
]

for sudoku in test_cases:
    assert solve(sudoku), "Sudoku should be solvable"
    assert is_valid_sudoku(sudoku), "Sudoku should be solved correctly"
