from easy21 import Easy21
import numpy as np
import matplotlib.pyplot as plot 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

"""
Monte-carlo control to solve easy21
"""

# states matrix for dealer and player
DEALER_STATE = 10 # 1 - 10
PLAYER_STATE = 21 # 1 - 21
ACTIONS = 2

# our number of counts
possible_states = np.zeros((DEALER_STATE, PLAYER_STATE), dtype=int)
possible_states_actions = np.zeros((DEALER_STATE, PLAYER_STATE, ACTIONS), dtype=int)

# our q values function
q_states_actions = np.zeros((DEALER_STATE, PLAYER_STATE, ACTIONS), dtype=np.float32)

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
    #initialize
    env = Easy21()
    runs = 5000

    for t in range(runs):
        env.start()
        results = []
        total_reward = 0.0
        start_dealer, start_player = env.state()
        print (" started at start state dealer %d and player %d" % (start_dealer, start_player))
        # run our episode
        while not env.is_finished():
            current_state = env.state()
            dealer, player = current_state
            print (" start at current state dealer %d, player %d" % (dealer, player))
            action = ep_greedy(current_state)
            print ("ep-greedy picked action %d" % action)
            next_state, reward = env.step(current_state, action)
            print ("------------------------------------")
            print (next_state)
            print (reward)
            print ("------------------------------------")
            # MC record our current state and the reward from this state onwards
            results.append((current_state, action))
            total_reward += reward
            if(dealer <= 10 and player <= 21):
                current_state = next_state

        for (state, action) in results:
            # incremet the state count
            dealer, player = state
            possible_states[dealer-1, player-1] += 1
            possible_states_actions[dealer-1, player-1, action] += 1
            # step-size
            alpha = 1/possible_states_actions[dealer-1, player-1, action]
            # update our value function in the direction towards the reward
            q_states_actions[dealer-1, player-1, action] += alpha * (total_reward - q_states_actions[dealer-1, player-1, action])

    with open('q_true.txt', 'w+') as outfile:
        outfile.write('# Array shape: {0}\n'.format(q_states_actions.shape))

        # equivalent to array[i,:,:]
        for data_slice in q_states_actions:
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            outfile.write('# New slice \n')
if __name__ == '__main__':
    main()