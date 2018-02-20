import numpy as np
import itertools
from IPython import embed


class GridWorld_MDP(object):
    '''
    Contains all methods pertinent to the Grid World's MDP including
    information on the states, the actions available in each state,
    the reward function, the transition probabilities, information
    on terminal & absorbing states.
    '''
    def __init__(self):
        # state information
        self.start = (2,0)
        self.blocked_tile = (1,1)
        self.terminal = [(0,3),(1,3)]
        self.absorbing = (-1,-1)
        all_locs = list(itertools.product(range(3),range(4))) + [self.absorbing]
        all_locs.remove(self.blocked_tile)
        self.num_states = len(all_locs) - 1

        self.loc2state = {loc:i for i, loc in enumerate(all_locs)}
        self.state2loc = {i:loc for i, loc in enumerate(all_locs)}

        # action information
        self.__action_map = {0: [-1,0], 1: [1,0], 2: [0,-1], 3: [0,1]}
        self.__action_noise = {0: [0,2,3], 1: [1,2,3], 2: [0,1,2], 3: [0,1,3]}
        self.action_str = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}
        self.num_actions = len(self.__action_map.keys())

    def is_terminal(self,s):
        '''
        Tests whether the state is a terminal state
        :param s: an integer representing the state
        :return: a boolean, True if s is a terminal state, else False
        '''
        return self.state2loc[s] in self.terminal

    def is_absorbing(self,s):
        '''
        Tests whether the state is an absorbing state
        :param s: an integer representing the state
        :return: a boolean, True if s is an absorbing state, else False
        '''
        return self.state2loc[s] == self.absorbing

    def S(self):
        '''
        Gives the set of states in the MDP
        :return: a list of integers representing each state
        '''
        return list(range(self.num_states))

    def A(self,s):
        '''
        Gives the actions available in each state
        :param s: an integer representing the state
        :return: a list of integers representing each action
        '''
        return self.__action_map.keys()

    def R(self,s):
        '''
        Gives the reward for taking an action in state s
        :param s: an integer representing the state
        :return: a float for the reward when acting in that state
        '''
        s = self.state2loc[s]
        if s == (0,3):
            return 1.
        elif s == (1,3):
            return -1.
        elif s == self.absorbing or s == self.blocked_tile:
            return 0.
        else:
            return -.04

    def P(self,snext,s,a):
        '''
        Gives the transition probability P(s'|s,a)
        :param snext: an integer representing the next state
        :param s: an integer representing the current state
        :param a: an integer representing the action
        :return: a float representing the probability of
        transitioning to state s' when taking action a in state s
        '''
        p_snexts = self.P_snexts(s,a)
        if snext in p_snexts:
            return p_snexts[snext]
        else:
            return 0.

    def P_snexts(self,s,a):
        '''
        Gives the transition probability distribution P(.|s,a)
        over all states with nonzero probability
        :param snext: an integer representing the next state
        :param s: an integer representing the current state
        :param a: an integer representing the action
        :return: a key-value dictionary with keys as integers
        representing the state s' and values as float representing
        the probability of transitioning to state s' when taking
        action a in state s. If s' does not appear as a key in
        the dictionary, then P(s'|s,a) = 0.
        '''
        if isinstance(s, int):
            s = self.state2loc[s]
        if a is None or s == self.absorbing or s == self.blocked_tile:
            return {self.loc2state[tuple(s)]: 1.}
        elif s in self.terminal:
            return {self.loc2state[tuple(self.absorbing)]: 1.}
        row, col = s
        p_snexts = {}
        for move in self.__action_noise[a]:
            snext = tuple(np.add(s,self.__action_map[move]))
            rownext, colnext = snext
            out_of_bounds = (rownext < 0 or rownext > 2 or colnext < 0 or colnext > 3)
            blocked = (snext == self.blocked_tile)
            forward = (move == a)
            if forward:
                p = 0.8
            else:
                p = 0.1
            if out_of_bounds or blocked:
                snext = tuple(s)
            if snext in p_snexts:
                p_snexts[snext] += p
            else:
                p_snexts[snext] = p
        return {self.loc2state[k]: v for k, v in p_snexts.items()}


class GridWorld(object):
    '''
    Contains all methods pertinent to the Grid World environment
    including reading the current state, performing actions,
    resetting the current state to the start or a random state,
    and testing for terminal or absorbing states.
    '''
    def __init__(self):
        # state information
        self.__start = (2,0)
        self.__blocked_tile = (1,1)
        self.__terminal = [(0,3),(1,3)]
        self.__absorbing = (-1,-1)
        all_locs = list(itertools.product(range(3),range(4))) + [self.__absorbing]
        all_locs.remove(self.__blocked_tile)
        self.num_states = len(all_locs) - 1

        self.loc2state = {loc:i for i, loc in enumerate(all_locs)}
        self.state2loc = {i:loc for i, loc in enumerate(all_locs)}
        
        all_locs.remove(self.__absorbing)
        self.__states = all_locs

        # action information
        self.__action_map = {0: [-1,0], 1: [1,0], 2: [0,-1], 3: [0,1]}
        self.__action_noise = {0: [0,2,3], 1: [1,2,3], 2: [0,1,2], 3: [0,1,3]}
        self.action_str = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}
        self.num_actions = len(self.__action_map.keys())

        # track aganets location
        self.__loc = self.__start

    def is_terminal(self):
        '''
        Tests whether the state is a terminal state
        :param s: an integer representing the state
        :return: a boolean, True if s is a terminal state, else False
        '''
        return self.__loc in self.__terminal

    def is_absorbing(self):
        '''
        Tests whether the state is an absorbing state
        :param s: an integer representing the state
        :return: a boolean, True if s is an absorbing state, else False
        '''
        return self.__loc == self.__absorbing

    def get_state(self):
        '''
        Gives the current state of the agent
        :return: an integer representing the agent's current state
        '''
        return self.loc2state[self.__loc]

    def reset_to_start(self):
        '''
        Resets the agents state to the start state
        :return: None
        '''
        self.__loc = self.__start

    def __R(self,s):
        '''
        Gives the reward for taking an action in state s
        :param s: a tuple (row,col) representing the state
        :return: a float for the reward when acting in that state
        '''
        if s == (0,3):
            return 1
        elif s == (1,3):
            return -1
        elif s == (-1,-1):
            return 0
        else:
            return -.04

    def __P_snexts(self,s,a):
        '''
        Gives the transition probability distribution P(.|s,a)
        over all states with nonzero probability
        :param s: an tuple representing the current state
        :param a: an integer representing the action
        :return: a key-value dictionary with keys as tuples
        representing the state s' and values as float representing
        the probability of transitioning to state s' when taking
        action a in state s. If s' does not appear as a key in
        the dictionary, then P(s'|s,a) = 0.
        '''
        if a is None or s == self.__absorbing or s == self.__blocked_tile:
            return {tuple(s): 1.}
        elif s in self.__terminal:
            return {tuple(self.__absorbing): 1.}
        row, col = s
        p_snexts = {}
        for move in self.__action_noise[a]:
            snext = tuple(np.add(s,self.__action_map[move]))
            rownext, colnext = snext
            out_of_bounds = (rownext < 0 or rownext > 2 or colnext < 0 or colnext > 3)
            blocked = (snext == self.__blocked_tile)
            forward = (move == a)
            if forward:
                p = 0.8
            else:
                p = 0.1
            if out_of_bounds or blocked:
                snext = tuple(s)
            if snext in p_snexts:
                p_snexts[snext] += p
            else:
                p_snexts[snext] = p
        return p_snexts

    def Act(self,a):
        '''
        Executes the action a for the agent and returns the
        corresponding reward for taking an action in state s
        :param a: an integer representing the action
        :return: a float for the reward when acting in that state
        '''
        s = self.__loc
        r = self.__R(s)
        p_snexts = self.__P_snexts(s,a)
        snexts, probs = list(zip(*list(p_snexts.items())))
        snext = list(snexts[self.__sample_cpt(probs)])
        self.__loc = tuple(snext)
        return r

    def __sample_cpt(self,probs):
        '''
        Draws a random sample from the probabilitiy distribution
        given by probs
        :param probs: a list of probabilities associated with each
        index. The probabilities must be between 0 and 1 and together
        must sum to 1.
        :return: an integer representing the index that was sampled
        '''
        z = np.random.rand()
        cdf = np.cumsum(probs)
        for i, thresh in enumerate(cdf):
            if z < thresh:
                return i



if __name__ == '__main__':
    mdp = GridWorld_MDP()
    env = GridWorld()
