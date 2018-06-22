from easy21 import Easy21
import numpy as np
import matplotlib.pyplot as plot 
from mpl_toolkits.mplot3d import Axes3D
"""
    TD lambda sarsa implementation to solve easy21
"""

# states matrix for dealer and player
DEALER_STATE = 10 # 1 - 10
PLAYER_STATE = 21 # 1 - 21
ACTIONS = 2

# LAMBDA 1 = MC
LAMBDA = 0.1
# Discount factor
GAMMA = 1

# our q values function
q_states_actions = np.zeros((DEALER_STATE, PLAYER_STATE, ACTIONS), dtype=np.float32)

# our number of counts
possible_states = np.zeros((DEALER_STATE, PLAYER_STATE), dtype=int)
possible_states_actions = np.zeros((DEALER_STATE, PLAYER_STATE, ACTIONS), dtype=int)

# eligibility traces
e_states_actions = np.zeros((DEALER_STATE, PLAYER_STATE, ACTIONS), dtype=np.float32)


def get_count(state):
    dealer, player = state
    # starts from 0 lol
    return possible_states[dealer-1, player-1]


def ep_greedy(state):
    """
        params:
            state - current game state
        
        return:
            action - a greedy action or a random action
    """
    n_0 = 100
    # number of time this state has been visited
    n_S = get_count(state)
    # ep is always <= 1
    ep = n_0 / (n_0 + n_S)

    # probability 1 - ep choose the greedy action
    # https://jamesmccaffrey.wordpress.com/2017/11/30/the-epsilon-greedy-algorithm/
    # basically if we have a random probability between 0 and 1
    # if this probability is bigger than our ep, then we choose 
    # greedy, else otherwise
    p = np.random.uniform(0, 1)
    if p > ep :
        # choose the action that gives us the best reward
        dealer, player = state
        return np.argmax(q_states_actions[dealer-1 , player-1, :])
    
    # choose our actions lol
    stick = np.random.uniform(0, 1) > 0.5
    return 0 if stick else 1

def main():
    # initialise our env
    env = Easy21()
    # run 1000 episodes
    runs = 1000
    
    global q_states_actions
    global e_states_actions
    global possible_states
    global possible_states_actions
    global GAMMA
    global LAMBDA

    # run our episodes
    for t in range(runs):
        # initialize our records
        results = []
        total_reward = 0.0
        # start the game
        env.start()
        #initialise first state and action
        start_state = env.state()  
        action = ep_greedy(start_state)

        # eligibility traces
        e_states_actions = np.zeros((DEALER_STATE, PLAYER_STATE, ACTIONS), dtype=np.float32)

        while not env.is_finished():
            current_state = env.state()
            # take action A, observe S' , Reward
            next_state, reward = env.step(current_state, action)
            next_dealer, next_player = next_state
            print ("next_dealer is %d, next_player is %d, immediate reward is %d" % (next_dealer, next_player, reward))
            current_dealer, current_player = current_state
            current_q_values = q_states_actions[current_dealer-1, current_player-1, action]
            # if we should still proceed
            if(next_dealer <= 10 and next_player <= 21):
                # choose A' from S' using e-greedy
                next_action = ep_greedy(next_state)

                # calculate the TD error (delta)
                next_q_values = q_states_actions[next_dealer-1, next_player-1, next_action]
                td_error = reward + (GAMMA * next_q_values) - current_q_values
            else:
                td_error = reward - current_q_values
            
            # add eligibility traces + 1
            e_states_actions[current_dealer-1, current_player-1, action] += 1

            # calculate step size online
            possible_states[current_dealer-1, current_player-1] += 1
            possible_states_actions[current_dealer-1, current_player-1, action] += 1

            # step-size
            alpha = 1/possible_states_actions[current_dealer-1, current_player-1, action]

            # update all action-value and eligilibity traces
            q_states_actions += alpha * td_error * e_states_actions
            e_states_actions = GAMMA * LAMBDA * e_states_actions

            if(next_dealer <= 10 and next_player <= 21):
                current_state = next_state
                action = next_action

    # Read the q true from disk
    q_true_txt = np.loadtxt('q_true.txt')

    # original shape of the array
    q_true = q_true_txt.reshape((DEALER_STATE,PLAYER_STATE,ACTIONS))
    q_diff = q_states_actions - q_true
    q_summed = np.sum(np.power(q_diff, 2))
    print (q_summed / (21 * 10 * 2))

if __name__ == '__main__':
    main()