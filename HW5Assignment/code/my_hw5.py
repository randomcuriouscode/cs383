from gridworld import GridWorld, GridWorld_MDP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

N = {x : y for (x, y) in ((i, 0) for i in range(11))}

def policy_evaluation(pi, mdp):
    '''
        Linear algebra policy evaluation for 1.1
    '''
    a = np.zeros((mdp.num_states,mdp.num_states))
    b = np.zeros(mdp.num_states)
    for s in mdp.S(): # s corresp to row of a
            b[s] = -mdp.R(s)

    #populate matrix rows

    for s in mdp.S(): # s corresp to row of a
        move = pi[s]
        nexts = mdp.P_snexts(s, move)

        a[s, s] -= 1 # subtract Ui(S) from RHS 

        for s_prime, probability in nexts.items(): # s_prime corresp to col of a
            if not mdp.is_absorbing(s_prime):
                a[s, s_prime] += probability
    ainv = np.linalg.inv(a)
    return np.dot(ainv, b)

def policy_evaluation_simpl(pi, U, mdp, gamma=1):
    '''
        Implementation of simple bellman update
    '''
    U_sol = np.zeros(mdp.num_states)

    for s in mdp.S(): # s corresp to row of a
        move = pi[s]
        nexts = mdp.P_snexts(s, move)
        U_sol[s] += mdp.R(s)

        for s_prime, probability in nexts.items(): # s_prime corresp to col of a
            if not mdp.is_absorbing(s_prime):
                U_sol[s] += (gamma * probability * U[s_prime])

    return U_sol

def policy_iteration_modified(mdp, gamma=1):
    '''
        Modified policy iteration for 1.1, computes U_0 with matrix math
        given pi_0 = 0. Computes U_1 with simplified bellman, then computes pi_1
        using U_1
    '''
    pi = [0 for x in range(mdp.num_states)]
    U = np.array([0 for x in range(mdp.num_states)])
    U = policy_evaluation(pi, mdp) #calculate U_1 using linalg
    print("_________________\nU_1:\n")
    for x in range(mdp.num_states):
        print("{} : {}".format(mdp.state2loc[x], U[x]))
    
    for s in mdp.S(): # iterate policy given new U_1
        #calculate rhs sum
        nexts = mdp.P_snexts(s, pi[s])
        rhs_sum = 0

        for s_prime, probability in nexts.items():
            if not mdp.is_absorbing(s_prime):
                rhs_sum += probability * U[s_prime]

        # create lhs mapping actions to sums
        lhs_dict = {}

        for a in mdp.A(s):
            nexts = mdp.P_snexts(s, a)

            lhs_sum = 0

            for s_prime, probability in nexts.items():
                if not mdp.is_absorbing(s_prime):
                    lhs_sum += probability * U[s_prime]
            lhs_dict[a] = lhs_sum


        #calculate max of lhs mapping
        lhs_sum = max(lhs_dict.items(), key=lambda k: k[1])
        # tuple (argmax, max)
        if lhs_sum[1] > rhs_sum:
            pi[s] = lhs_sum[0]

    print([mdp.action_str[x] for x in pi])

    U = policy_evaluation_simpl(pi, U, mdp, gamma)
    print("_________________\nU_2:\n")

    for x in range(mdp.num_states):
        print("{} : {}".format(mdp.state2loc[x], U[x]))
    
    return pi

def policy_iteration(mdp, gamma=1, iters=5, plot=True):
    '''
    Performs policy iteration on an mdp and returns the value function and policy 
    :param mdp: mdp class (GridWorld_MDP) 
    :param gam: discount parameter should be in (0, 1] 
    :param iters: number of iterations to run policy iteration for
    :param plot: boolean for if a plot should be generated for the utilities of the start state
    :return: two numpy arrays of length |S| and one of length iters.
    Two arrays of length |S| are U and pi where U is the value function
    and pi is the policy. The third array contains U(start) for each iteration
    the algorithm.
    '''
    pi = np.zeros(mdp.num_states, dtype=np.int)
    U = np.zeros(mdp.num_states)
    Ustart = []

    for x in range(iters):
        Ustart.append(U[mdp.loc2state[mdp.start]])
        U = policy_evaluation(pi, mdp)
        for s in mdp.S():
            #calculate rhs sum
            nexts = mdp.P_snexts(s, pi[s])
            rhs_sum = 0

            for s_prime, probability in nexts.items():
                if not mdp.is_absorbing(s_prime):
                    rhs_sum += probability * U[s_prime]

            # create lhs mapping actions to sums
            lhs_dict = {}

            for a in mdp.A(s):
                nexts = mdp.P_snexts(s, a)

                lhs_sum = 0

                for s_prime, probability in nexts.items():
                    if not mdp.is_absorbing(s_prime):
                        lhs_sum += probability * U[s_prime]
                lhs_dict[a] = lhs_sum


            #calculate max of lhs mapping
            lhs_sum = max(lhs_dict.items(), key=lambda k: k[1])
            # tuple (argmax, max)
            if lhs_sum[1] > rhs_sum:
                pi[s] = lhs_sum[0]


    if plot:
        fig = plt.figure()
        plt.title("Policy Iteration with $\gamma={0}$".format(gamma))
        plt.xlabel("Iteration (k)")
        plt.ylabel("Utility of Start")
        plt.ylim(-1, 1)
        plt.plot(Ustart)

        pp = PdfPages('./plots/piplot.pdf')
        pp.savefig(fig)
        plt.close()
        pp.close()

    #U and pi should be returned with the shapes and types specified
    return U, pi, np.array(Ustart)


def td_update(v, s1, r, s2, terminal, alpha, gamma):
    '''
    Performs the TD update on the value function v for one transition (s,a,r,s').
    Update to v should be in place.
    :param v: The value function, a numpy array of length |S|
    :param s1: the current state, an integer 
    :param r: reward for the transition
    :param s2: the next state, an integer
    :param terminal: bool for if the episode ended
    :param alpha: learning rate parameter
    :param gamma: discount factor
    :return: Nothing
    '''
    #TODO implement the TD Update
    #you should update the value function v inplace (does not need to be returned)
    if not terminal:
        v[s1] = v[s1] + alpha * (r + (gamma * v[s2]) - v[s1])

def td_episode(env, pi, v, gamma, alpha, max_steps=1000):
    '''
    Agent interacts with the environment for one episode update the value function after
    each iteration. The value function update should be done with the TD learning rule.
    :param env: environment object (GridWorld)
    :param pi: numpy array of length |S| representing the policy
    :param v: numpy array of length |S| representing the value function
    :param gamma: discount factor
    :param alpha: learning rate
    :param max_steps: maximum number of steps in the episode
    :return: two floats G, v0 where G is the discounted return and v0 is the value function of the initial state (before learning)
    '''
    G = 0.
    v0 = 0.

    #TODO implement the agent interacting with the environment for one episode
    # episode ends when max_steps have been completed
    # episode ends when env is in the absorbing state
    # Learning should be done online (after every step)
    # return the discounted sum of rewards G, and the value function's estimate from the initial state v0
    # the value function estimate should be before any learn takes place in this episode

    global N
    env.reset_to_start()

    s = None
    a = None
    r = None

    start_state = env.get_state()
    start_action = pi[start_state]

    v0 = v[start_state]

    for x in range(max_steps): # episode ends when max_steps have been completed
        if x == 0: # first iteration use start state
            r_prime = env.Act(start_action)
            s_prime = env.get_state()
        else:
            r_prime = env.Act(a) # sample s', r'
            s_prime = env.get_state()

        if not env.is_absorbing() and N[s_prime] == 0:
            v[s_prime] = r_prime
        elif env.is_absorbing():
            v[s] = r_prime

        if s is not None:
            N[s] += 1
            td_update(v, s, r, s_prime, env.is_absorbing(), alpha, gamma)

        if env.is_absorbing():
            G += (gamma ** x) * r_prime
            return G, v0
        else:
            s = s_prime
            a = pi[s_prime]
            r = r_prime
            G += (gamma ** x) * r

    return G, v0

def td_learning(env, pi, gamma, alpha, episodes=200, plot=True):
    '''
    Evaluates the policy pi in the environment by estimating the value function
    with TD updates  
    :param env: environment object (GridWorld)
    :param pi: numpy array of length |S|, representing the policy 
    :param gamma: discount factor
    :param alpha: learning rate
    :param episodes: number of episodes to use in evaluating the policy
    :param plot: boolean for if a plot should be generated for returns and estimates
    :return: Two lists containing the returns for each episode and the value function estimates, also returns the value function
    '''
    returns, estimates = [], []
    v = np.zeros(env.num_states)

    # TODO Implement the td learning for every episode
    # value function should start at 0 for all states
    # return the list of returns, and list of estimates for all episodes
    # also return the value function v

    global N
    N = {x : y for (x, y) in ((i, 0) for i in range(env.num_states))}

    for x in range(episodes):
        G, v0 = td_episode(env, pi, v, gamma, alpha)
        returns.append(G)
        estimates.append(v0)

    if plot:
        fig = plt.figure()
        plt.title("TD Learning with $\gamma={0}$ and $\\alpha={1}$".format(gamma, alpha))
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.ylim(-4, 1)
        plt.plot(returns)
        plt.plot(estimates)
        plt.legend(['Returns', 'Estimate'])

        pp = PdfPages('./plots/tdplot.pdf')
        pp.savefig(fig)
        plt.close()
        pp.close()

    return returns, estimates, v

def egreedy(q, s, eps):
    '''
    Epsilon greedy action selection for a discrete Q function.
    :param q: numpy array of size |S|X|A| representing the state action value look up table
    :param s: the current state to get an action (an integer)
    :param eps: the epsilon parameter to randomly select an action
    :return: an integer representing the action
    '''

    # TODO implement epsilon greedy action selection

    sample = np.random.random_sample() #gen number from 1-1000
    action = None

    if sample < eps: 
        action = np.random.randint(low=0, high=4) #gen int from 0-3
    else:
        #find argmax_a of q(s,a)
        action = np.argmax(q[s])

    return action

def q_update(q, s1, a, r, s2, terminal, alpha, gamma):
    '''
    Performs the Q learning update rule for a (s,a,r,s') transition. 
    Updates to the Q values should be done inplace
    :param q: numpy array of size |S|x|A| representing the state action value table
    :param s1: current state
    :param a: action taken
    :param r: reward observed
    :param s2: next state
    :param terminal: bool for if the episode ended
    :param alpha: learning rate
    :param gamma: discount factor
    :return: None
    '''

    # TODO implement Q learning update rule
    # update should be done inplace (not returned)

    if not terminal:
        q[s1,a] = q[s1,a] + (alpha * (r + gamma * max(q[s2]) - q[s1,a]))

def q_episode(env, q, eps, gamma, alpha, max_steps=1000):
    '''
    Agent interacts with the environment for an episode update the state action value function
    online according to the Q learning update rule. Actions are taken with an epsilon greedy policy
    :param env: environment object (GridWorld)
    :param q: numpy array of size |S|x|A| for state action value function
    :param eps: epsilon greedy parameter
    :param gamma: discount factor
    :param alpha: learning rate
    :param max_steps: maximum number of steps to interact with the environment
    :return: two floats: G, q0 which are the discounted return and the estimate of the return from the initial state
    '''
    G = 0.
    q0 = 0.

    env.reset_to_start()

    s = None
    a = None
    r = None

    start_state = env.get_state()
    start_action = egreedy(q,start_state, eps)

    q0 = max(q[start_state])

    for x in range(max_steps): # episode ends when max_steps have been completed
        if x == 0: # first iteration use start state
            r_prime = env.Act(start_action)
            s_prime = env.get_state()
        else:
            r_prime = env.Act(a) # sample s', r'
            s_prime = env.get_state()

        if env.is_absorbing():
            q[s].fill(r_prime)

        if s is not None:
            q_update(q, s, a, r, s_prime, env.is_absorbing(), alpha, gamma)

        if env.is_absorbing():
            G += (gamma ** x) * r_prime
            return G, q0
        else:
            s = s_prime
            a = egreedy(q, s, eps)
            r = r_prime
            G += (gamma ** x) * r

    return G, q0

def q_learning(env, eps, gamma, alpha, episodes=200, plot=True):
    '''
    Learns a policy by estimating the state action values through interactions 
    with the environment.  
    :param env: environment object (GridWorld)
    :param eps: epsilon greedy action selection parameter
    :param gamma: discount factor
    :param alpha: learning rate
    :param episodes: number of episodes to learn
    :param plot: boolean for if a plot should be generated returns and estimates
    :return: Two lists containing the returns for each episode and the action value function estimates of the return, also returns the Q table
    '''
    returns, estimates = [], []
    q = np.zeros((env.num_states, env.num_actions))

    # TODO implement Q learning over episodes
    # return the returns and estimates for each episode and the Q table

    for x in range(episodes):
        G, q0 = q_episode(env, q, eps, gamma, alpha)
        returns.append(G)
        estimates.append(q0)

    if plot:
        fig = plt.figure()
        plt.title("Q Learning with $\gamma={0}$, $\epsilon={1}$, and $\\alpha={2}$".format(gamma, eps, alpha))
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.ylim(-4, 1)
        plt.plot(returns)
        plt.plot(estimates)
        plt.legend(['Returns', 'Estimate'])

        pp = PdfPages('./plots/qplot.pdf')
        pp.savefig(fig)
        plt.close()
        pp.close()

    return returns, estimates, q



if __name__ == '__main__':
    env = GridWorld()
    mdp = GridWorld_MDP()

    U, pi, Ustart = policy_iteration(mdp, plot=True)
    print(pi)
    for x in range(env.num_states):
        print("{} : {}".format(env.state2loc[x], U[x]))
    print("_________________")
    vret, vest, v = td_learning(env, pi, gamma=1., alpha=0.1, episodes=2000, plot=True)
    for x in range(env.num_states):
        print("{} : {}".format(env.state2loc[x], v[x]))
    qret, qest, q = q_learning(env, eps=0.1, gamma=1., alpha=0.1, episodes=20000, plot=True)
    iteration = 0
    pi_ret = np.zeros(env.num_states)
    for x in range(env.num_states):
        pi_ret[x] = np.argmax(q[x])
    print("Iteration: {} ______________".format(iteration))
    print(pi_ret)

    for x in range(env.num_states):
        print("{} : {}".format(env.state2loc[x], max(q[x])))

    while not np.array_equal(pi_ret, pi):
        iteration += 1
        qret, qest, q = q_learning(env, eps=0.1, gamma=1., alpha=0.1, episodes=20000, plot=True)
        pi_ret = np.zeros(env.num_states)
        for x in range(env.num_states):
            pi_ret[x] = np.argmax(q[x])
        print("Iteration: {} ______________".format(iteration))
        print(pi_ret)

        for x in range(env.num_states):
            print("{} : {}".format(env.state2loc[x], max(q[x])))
