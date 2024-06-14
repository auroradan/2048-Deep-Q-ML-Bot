import numpy as np
from Direction import Direction

class Board:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.add_num()
        self.add_num()

    def add_num(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = empty_cells[np.random.choice(len(empty_cells))]
            self.board[x][y] = 2 if np.random.rand() < 0.9 else 4

    def move(self, direction):
        original_board = self.board.copy()
        if direction == Direction.NORTH:
            self.board = self.move_north()
        elif direction == Direction.SOUTH:
            self.board = self.move_south()
        elif direction == Direction.EAST:
            self.board = self.move_east()
        elif direction == Direction.WEST:
            self.board = self.move_west()
        if not np.array_equal(original_board, self.board):
            self.add_num()

    def compress(self, grid):
        new_grid = np.zeros_like(grid)
        for row in range(4):
            pos = 0
            for col in range(4):
                if grid[row][col] != 0:
                    new_grid[row][pos] = grid[row][col]
                    pos += 1
        return new_grid

    def merge(self, grid):
        for row in range(4):
            for col in range(3):
                if grid[row][col] == grid[row][col + 1] and grid[row][col] != 0:
                    grid[row][col] *= 2
                    grid[row][col + 1] = 0
        return grid

    def move_north(self):
        rotated_board = np.rot90(self.board, -1)
        compressed_board = self.compress(rotated_board)
        merged_board = self.merge(compressed_board)
        final_board = self.compress(merged_board)
        return np.rot90(final_board, 1)

    def move_south(self):
        rotated_board = np.rot90(self.board, 1)
        compressed_board = self.compress(rotated_board)
        merged_board = self.merge(compressed_board)
        final_board = self.compress(merged_board)
        return np.rot90(final_board, -1)

    def move_east(self):
        flipped_board = np.fliplr(self.board)
        compressed_board = self.compress(flipped_board)
        merged_board = self.merge(compressed_board)
        final_board = self.compress(merged_board)
        return np.fliplr(final_board)

    def move_west(self):
        compressed_board = self.compress(self.board)
        merged_board = self.merge(compressed_board)
        final_board = self.compress(merged_board)
        return final_board

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for direction in Direction:
            test_board = self.board.copy()
            if direction == Direction.NORTH:
                test_board = self.move_north()
            elif direction == Direction.SOUTH:
                test_board = self.move_south()
            elif direction == Direction.EAST:
                test_board = self.move_east()
            elif direction == Direction.WEST:
                test_board = self.move_west()
            if not np.array_equal(test_board, self.board):
                return False
        return True

    def get_state(self):
        return self.board
