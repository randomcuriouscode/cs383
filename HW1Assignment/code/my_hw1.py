from eight_puzzle import Puzzle
from time import time
from math import fabs
from heapq import heappush, heappop, heapify
import queue
from itertools import count

##################################################################
### Node class and helper functions provided for your convenience.
### DO NOT EDIT!
##################################################################
class Node:
    """
    A class representing a node.
    - 'state' holds the state of the node.
    - 'parent' points to the node's parent.
    - 'action' is the action taken by the parent to produce this node.
    - 'path_cost' is the cost of the path from the root to this node.
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def gen_child(self, problem, action):
        """
        Returns the child node resulting from applying 'action' to this node.
        """
        return Node(state=problem.transitions(self.state, action),
                    parent=self,
                    action=action,
                    path_cost=self.path_cost + problem.step_cost(self.state, action))

    @property
    def state_hashed(self):
        """
        Produces a hashed representation of the node's state for easy
        lookup in a python 'set'.
        """
        return hash(str(self.state))

##################################################################
### Node class and helper functions provided for your convenience.
### DO NOT EDIT!
##################################################################
def retrieve_solution(node,num_explored,num_generated):
    """
    Returns the list of actions and the list of states on the
    path to the given goal_state node. Also returns the number
    of nodes explored and generated.
    """
    actions = []
    states = []
    while node.parent is not None:
        actions += [node.action]
        states += [node.state]
        node = node.parent
    states += [node.state]
    return actions[::-1], states[::-1], num_explored, num_generated

##################################################################
### Node class and helper functions provided for your convenience.
### DO NOT EDIT!
##################################################################
def print_solution(solution):
    """
    Prints out the path from the initial state to the goal given
    a tuple of (actions,states) corresponding to the solution.
    """
    actions, states, num_explored, num_generated = solution
    print('Start')
    for step in range(len(actions)):
        print(puzzle.board_str(states[step]))
        print()
        print(actions[step])
        print()
    print('Goal')
    print(puzzle.board_str(states[-1]))
    print()
    print('Number of steps: {:d}'.format(len(actions)))
    print('Nodes explored: {:d}'.format(num_explored))
    print('Nodes generated: {:d}'.format(num_generated))


################################################################
### Skeleton code for your Astar implementation. Fill in here.
################################################################
class Astar:
    """
    A* search.
    - 'problem' is a Puzzle instance.
    """
    

    def __init__(self, problem):
        self.problem = problem

    def solve(self):
        """
        Perform A* search and return a solution using `retrieve_solution'
        (if a solution exists).
        IMPORTANT: Use node generation time (i.e., time.time()) to split
        ties among nodes with equal f(n).
        """

        frontier = queue.PriorityQueue()
        explored = set()
        num_explored = 0
        num_generated = 1
        counter = count()
        frontier_map = {} # mapping from state_hashed to entry

        def pop_from_frontier():
            priority, time, node = frontier.get()
            del frontier_map[node[0].state_hashed]
            return (priority, time, node)
            raise KeyError('pop from an empty priority queue')

        def add_to_frontier(item):
            frontier_map[item[2][0].state_hashed] = item
            frontier.put(item)

        def replace_in_frontier(item):
            #replace existing priority, item with new
            existing = frontier_map[item[2][0].state_hashed]
            existing[0][0] = item[0][0]
            existing[2][0] = item[2][0]
            heapify(frontier.queue)

        # init start node
        start_node = Node(self.problem.init_state, None, None, 0)

        add_to_frontier(([self.f(start_node)], [next(counter)], [start_node]))

        while not frontier.empty():
            cur_item = pop_from_frontier()

            if self.problem.is_goal(cur_item[2][0].state):
                #print(len(frontier_map))
                return retrieve_solution(cur_item[2][0], num_explored, num_generated)

            explored.add(cur_item[2][0].state_hashed)

            num_explored += 1

            for action in self.problem.actions(cur_item[2][0].state):
                child = cur_item[2][0].gen_child(self.problem, action)

                num_generated += 1

                if child.state_hashed in explored:
                    continue

                if child.state_hashed not in frontier_map:
                    add_to_frontier(([self.f(child)], [next(counter)], [child]))

                elif self.f(child) < frontier_map[child.state_hashed][0][0]:
                    #print("child.h:{},child.f:{},frontier.f:{}".format(self.h(child),self.f(child), frontier_map[child.state_hashed][0][0]))
                    #print("Replaced node with path cost {} with {}".format(frontier_map[child.state_hashed][0][0], self.f(child)))
                    replace_in_frontier(([self.f(child)], [next(counter)], [child]))
        return None

    def unflatten(flat_state):
        '''
        Unflattens a flat state list
        '''
        return [flat_state[i : i + 3] for i in range(0, len(flat_state), 3)]        

    def flatten(unflat_state):
        '''
        Flattens a unflat state list
        '''
        return [unflat_state[x][y] for x in range(0, 3) for y in range(0,3)]

    def f(self,node):
        '''
        Returns a lower bound estimate on the cost from root through node
        to the goal.
        '''
        return node.path_cost + self.h(node)

    def h(self,node):
        '''
        Returns a lower bound estimate on the cost from node to the goal
        using the Manhattan distance heuristic.
        '''

        unflat_init = Astar.unflatten(node.state) 
        unflat_goal = Astar.unflatten(self.problem.goal_state) 

        def search_goal(number):
            t = [i for i in unflat_goal if number in i][0]
            return unflat_goal.index(t),t.index(number)

        h_val = 0

        for x in range(len(unflat_goal)):
            for y in range(len(unflat_goal[0])):
                node_val = unflat_init[x][y]
                goal_val = unflat_goal[x][y]

                if node_val == goal_val or node_val == 0:
                    continue

                expected_x, expected_y = search_goal(node_val)

                delta = fabs(expected_x - x) + fabs(expected_y - y)
                h_val += delta

        return int(h_val)

    def branching_factor(self, board, trials=100):
        '''
        Returns an average upper bound for the effective branching factor.
        '''
        b_hi = 0.0  # average upper bound for branching factor
        for t in range(trials):
            puzzle = Puzzle(board).shuffle()
            solver = Astar(puzzle)
            actions, states, num_explored, num_generated = solver.solve()
            b_hi += float(num_generated) ** (1 / len(states))
        return b_hi / trials


if __name__ == '__main__':
    # Simple puzzle test
    board = [[3,1,2],
             [4,0,5],
             [6,7,8]]

    puzzle = Puzzle(board)
    solver = Astar(puzzle)
    solution = solver.solve()
    print_solution(solution)

    t1 = time()

    # Harder puzzle test
    board = [[7,2,4],
             [5,0,6],
             [8,3,1]]

    puzzle = Puzzle(board)
    solver = Astar(puzzle)
    solution = solver.solve()
    t2 = time()
    print_solution(solution)

    print("t2 - t1: {}".format(t2-t1))

    # branching factor test
    t1 = time()
    b_hi = solver.branching_factor(board, trials=100)
    t2 = time()
    print('Upper bound on effective branching factor: {:.2f}'.format(b_hi))
    print("t2 - t1: {}".format(t2-t1))

    