from sudoku import Sudoku
from copy import deepcopy
from collections import deque
from time import time


class CSP_Solver(object):
    """
    This class is used to solve the CSP with backtracking.
    """
    def __init__(self, puzzle_file):
        self.sudoku = Sudoku(puzzle_file)
        self.guesses = 0
        self.domain = deepcopy(self.sudoku.board)
        self.arcs = []
        self.unassigned = deque()
        self.assigned = {}

        for row in range(9):
            for col in range(9):
                val = self.sudoku.board[row][col]
                if val == 0: # if val is empty, domain is every possible number
                    self.domain[row][col] = [x for x in range(1,10)]
                    self.unassigned.append((row,col))
                else: 
                    self.domain[row][col] = val
                    self.assigned[(row,col)] = val

    ################################################################
    ### YOU MUST EDIT THIS FUNCTION!!!!!
    ### We will test your code by constructing a csp_solver instance
    ### e.g.,
    ### csp_solver = CSP_Solver('puz-001.txt')
    ### solved_board, num_guesses = csp_solver.solve()
    ### so your `solve' method must return these two items.
    ################################################################
    def solve(self):
        """
        Solves the Sudoku CSP and returns a list of lists representation
        of the solved sudoku puzzle as well as the number of guesses
        (assignments) required to solve the problem.
        YOU MUST EDIT THIS FUNCTION!!!!!
        """
        self.backtracking_search()

        print(self.sudoku.board_str())

        return self.sudoku.board, self.guesses

    def consistent(self, var, value):
        # check if given (x,y) is consistent with current board

        # check row consistency

        for col in [x for x in range(9) if x != var[1]]:
            if self.sudoku.board[var[0]][col] == value:
                return False

        # check col consistency

        for row in [x for x in range(9) if x != var[0]]:
            if self.sudoku.board[row][var[1]] == value:
                return False

        # check box consistency

        # define starting indices of box for var

        start_row = var[0]//3 * 3
        start_col = var[1]//3 * 3

        for row in [x for x in range(start_row, start_row + 3) if x != var[0]]:
            for col in [x for x in range(start_col, start_col + 3) if x != var[1]]:
                if self.sudoku.board[row][col] == value:
                    return False

        return True

    def select_unassigned_var(self, assignment):
        for row in range(9):
            for col in range(9):
                val = self.sudoku.board[row][col]
                if val == 0: # if val is empty, domain is every possible number
                    return (row,col)

    def order_domain_values(self, var, assignment):
        return self.domain[var[0]][var[1]]

    def backtracking_search(self):
        return self.recursive_backtracking(self.assigned)

    def recursive_backtracking(self, assignment):
        if self.sudoku.complete():
            return assignment
        var = self.select_unassigned_var(assignment)
        for value in self.order_domain_values(var, assignment):
            self.guesses += 1
            if self.consistent(var, value):
                assignment[var] = value
                self.sudoku.board[var[0]][var[1]] = value
                result = self.recursive_backtracking(assignment)
                if result is not None:
                    return result
                assignment.pop(var) # get rid of assignment
                self.sudoku.board[var[0]][var[1]] = 0
        return None

class CSP_Solver_MRV(object):
    """
    This class is used to solve the CSP with backtracking.
    """
    def __init__(self, puzzle_file):
        self.sudoku = Sudoku(puzzle_file)
        self.guesses = 0
        self.domain = deepcopy(self.sudoku.board)
        self.arcs = []
        self.unassigned = deque()
        self.assigned = {}

        for row in range(9):
            for col in range(9):
                val = self.sudoku.board[row][col]
                if val == 0: # if val is empty, domain is every possible number
                    self.domain[row][col] = [x for x in range(1,10)]
                    self.unassigned.append((row,col))
                else: 
                    self.domain[row][col] = val
                    self.assigned[(row,col)] = val

    ################################################################
    ### YOU MUST EDIT THIS FUNCTION!!!!!
    ### We will test your code by constructing a csp_solver instance
    ### e.g.,
    ### csp_solver = CSP_Solver('puz-001.txt')
    ### solved_board, num_guesses = csp_solver.solve()
    ### so your `solve' method must return these two items.
    ################################################################
    def solve(self):
        """
        Solves the Sudoku CSP and returns a list of lists representation
        of the solved sudoku puzzle as well as the number of guesses
        (assignments) required to solve the problem.
        YOU MUST EDIT THIS FUNCTION!!!!!
        """
        self.backtracking_search()

        print(self.sudoku.board_str())

        return self.sudoku.board, self.guesses

    def consistent(self, var, value):
        # check if given (x,y) is consistent with current board

        # check row consistency

        for col in [x for x in range(9) if x != var[1]]:
            if self.sudoku.board[var[0]][col] == value:
                return False

        # check col consistency

        for row in [x for x in range(9) if x != var[0]]:
            if self.sudoku.board[row][var[1]] == value:
                return False

        # check box consistency

        # define starting indices of box for var

        start_row = var[0]//3 * 3
        start_col = var[1]//3 * 3

        for row in [x for x in range(start_row, start_row + 3) if x != var[0]]:
            for col in [x for x in range(start_col, start_col + 3) if x != var[1]]:
                if self.sudoku.board[row][col] == value:
                    return False

        return True

    def free_vals(self, var):
        # given unassigned var, return number of free values in domain
        # and set domain for that value

        # go through row

        tempdomain = [x for x in range(1,10)]

        for col in [x for x in range(9) if x != var[1]]:
            if self.sudoku.board[var[0]][col]!= 0 and \
                self.sudoku.board[var[0]][col] in tempdomain:
                tempdomain.remove(self.sudoku.board[var[0]][col])

        # go through col

        for row in [x for x in range(9) if x != var[0]]:
            if self.sudoku.board[row][var[1]]!= 0 and  \
                self.sudoku.board[row][var[1]] in tempdomain:
                tempdomain.remove(self.sudoku.board[row][var[1]])

        # go through box

        # define starting indices of box for var

        start_row = var[0]//3 * 3
        start_col = var[1]//3 * 3

        for row in [x for x in range(start_row, start_row + 3) if x != var[0]]:
            for col in [x for x in range(start_col, start_col + 3) if x != var[1]]:
                if self.sudoku.board[row][col] != 0 and \
                    self.sudoku.board[row][col] in tempdomain:
                    tempdomain.remove(self.sudoku.board[row][col])

        return len(tempdomain), tempdomain

    def select_unassigned_var(self, assignment):
        """
        for row in range(9):
            for col in range(9):
                val = self.sudoku.board[row][col]
                if val == 0: # if val is empty, domain is every possible number
                    return (row,col)
        """

        leastval = None
        leastdomain = []
        nleast = 9 

        for row in range(9):
            for col in range(9):
                val = self.sudoku.board[row][col]
                if val == 0 :
                    fv, ld = self.free_vals((row,col))
                    if fv < nleast:
                        nleast = fv
                        leastval = (row,col)
                        leastdomain = ld 

        return leastval, leastdomain

    def backtracking_search(self):
        return self.recursive_backtracking(self.assigned)

    def recursive_backtracking(self, assignment):
        if self.sudoku.complete():
            return assignment
        var, ld = self.select_unassigned_var(assignment)
        
        for value in ld:
            self.guesses += 1
            if self.consistent(var, value):
                assignment[var] = value
                self.sudoku.board[var[0]][var[1]] = value
                result = self.recursive_backtracking(assignment)
                if result is not None:
                    return result
                assignment.pop(var) # get rid of assignment
                self.sudoku.board[var[0]][var[1]] = 0
        return None


if __name__ == '__main__':
    csp_solver_mrv = CSP_Solver_MRV('puz-001.txt')
    print(csp_solver_mrv.solve())
    #csp_solver_mrv.sudoku.write('puz-001-solved-mrv.txt')

    csp_solver_mrv = CSP_Solver_MRV('puz-026.txt')
    print(csp_solver_mrv.solve())
    #csp_solver_mrv.sudoku.write('puz-026-solved-mrv.txt')

    csp_solver_mrv = CSP_Solver_MRV('puz-051.txt')
    print(csp_solver_mrv.solve())
    #csp_solver_mrv.sudoku.write('puz-051-solved-mrv.txt')

    csp_solver_mrv = CSP_Solver_MRV('puz-076.txt')
    print(csp_solver_mrv.solve())
    #csp_solver_mrv.sudoku.write('puz-076-solved-mrv.txt')

    csp_solver_mrv = CSP_Solver_MRV('puz-090.txt')
    print(csp_solver_mrv.solve())
    #csp_solver_mrv.sudoku.write('puz-090-solved-mrv.txt')

    csp_solver_mrv = CSP_Solver_MRV('puz-100.txt')
    print(csp_solver_mrv.solve())
    #csp_solver_mrv.sudoku.write('puz-100-solved-mrv.txt')
