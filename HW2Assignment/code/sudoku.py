class Sudoku(object):
    """
    A class for basic Sudoku functionality.
    - 'puzzle' should either be a filename for the puzzle to load from the 'puzzles/' folder
       or a list of lists sudoku board with entries as ints and empty tiles represented by 0
       e.g., problem = Sudoku('puz-001.txt')
       or
       board = [[7, 8, 1, 6, 0, 2, 9, 0, 5],
                [9, 0, 2, 7, 1, 0, 0, 0, 0],
                [0, 0, 6, 8, 0, 0, 0, 1, 2],
                [2, 0, 0, 3, 0, 0, 8, 5, 1],
                [0, 7, 3, 5, 0, 0, 0, 0, 4],
                [0, 0, 8, 0, 0, 9, 3, 6, 0],
                [1, 9, 0, 0, 0, 7, 0, 8, 0],
                [8, 6, 7, 0, 0, 3, 4, 0, 9],
                [0, 0, 5, 0, 0, 0, 1, 0, 0]]
       e.g., problem = Sudoku(board)
    """

    EMPTY = 0

    def __init__(self, puzzle):
        """
        Constructs the Sudoku class with the given puzzle. See description of Sudoku
        class for which arguments to pass to Sudoku(puzzle).
        self.board is a list of lists representation of the board using ints - you
            can update this board with your new moves
        self.orig_board is a list of lists representation of the original board
        """
        if isinstance(puzzle,str):
            self.board = self.load_board(puzzle)
        else:
            self.board = puzzle
        self.orig_board = self.board

    def load_board(self, puzzle_file):
        """
        Loads a puzzle txt file and converts it to a list of lists integer
        representation with empty tiles as 0.
        """
        with open('puzzles/'+puzzle_file, 'r') as f:
            board = []
            for line in f:
                row = [int(s) for s in line.replace('-',str(Sudoku.EMPTY)).split(' ')]
                board += [row]
        return board

    def write(self, filename):
        """
        Writes the board to file "filename".
        """
        with open('solved_mrv/'+filename, 'w') as f:
            f.write(self.board_str())

    def board_str(self):
        """
        Returns a string representation of the board for pretty printing to screen.
        """
        out = ''
        for line in self.board:
            str_line = [str(i) if i!=Sudoku.EMPTY else '-' for i in line]
            out += ' '.join(str_line)+'\n'
        return out[:-1]

    def complete(self):
        """
        Tests whether all the tiles in the board are filled in.
        Returns true if the board is filled. False, otherwise.
        """
        return all([all(row) for row in self.board])

    def overwritten(self):
        """
        Tests whether one of the original tiles was overwritten. You should NOT
        be overwriting any of the original tiles, so hopefully this returns False.
        Returns True if the board was overwritten. False, otherwise.
        """
        for row, line in enumerate(self.orig_board):
            for col, num in enumerate(line):
                if num != Sudoku.EMPTY:
                    if num != self.board[row][col]:
                        return True
        return False


if __name__ == '__main__':
    problem = Sudoku('puz-001.txt')
    problem.write('puz-001-solved.txt')
    print(problem.board_str())
    print(problem.complete())
    print(problem.overwritten())
    problem2 = Sudoku(problem.board)
    print(help(Sudoku))
