from easy21 import Easy21
import numpy as np
"""
    linear function approximation for action-value to solve easy21
"""

DEALERS_FEATURE = 3 # 1-4, 4-7, 7-10
PLAYERS_FEATURE = 6 # 1-6, 4-9, 7-12, 10-15, 13-18, 16-21
ACTIONS_FEATURE = 2 # hit, stick

EPSILON = 0.05
ALPHA = 0.01
GAMMA = 1
DEALERS_STATE = 10
PLAYERS_STATE = 21

q_states_actions = np.zeros((DEALERS_STATE, PLAYERS_STATE, ACTIONS_FEATURE), dtype=np.float32)

def ep_greedy(state):
    """
        params:
            state - current game state
        
        return:
            action - a greedy action or a random action
    """
    # ep is always <= 1
    ep = EPSILON

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

def Features(dealer, player, action):
    """
    will be like (<1-4>, <1-6>, <hit>)
                 (<1-4>, <4-9>, <hit>)
                 (<1-4>, <7-12>, <hit>)
                 to
                 (<7-10>,<16-21>,<stick>)

    Params:
        dealer - int 
        player - int
        action - int
    returns:
        a vector of zeros and ones to indicate which feature it is at.
    """
    # initialize feature vectors as zero
    features = np.zeros(DEALERS_FEATURE * PLAYERS_FEATURE * ACTIONS_FEATURE)

    for dealer_index, (lower, upper) in enumerate(zip(range(1,8, 3), range(4,11,3))):
        if lower <= dealer <= upper:
            for player_index, (lower, upper) in enumerate(zip(range(1,17,3), range(6, 22, 3))):
                if lower <= player <= upper:
                    for a in range(ACTIONS_FEATURE):
                        if a==action:
                            dealer_f = dealer_index+1
                            player_f = player_index+1
                            action_f = a * 18
                            index = (dealer_f * player_f + action_f) - 1
                            features[index] = 1
    
    return features
    

def Q_approx(dealer, player, action, weights):
    # linear weighted sum
    value = np.dot(Features(dealer, player, action), weights)
    q_states_actions[dealer, player, action] = value
    return value

def main():
    # initialise our env
    env = Easy21()
    # run 1000 episodes
    runs = 1000

    global q_states_actions
    global DEALERS_FEATURE
    global PLAYERS_FEATURE
    global ACTIONS_FEATURE
    
    for x in [x * 0.1 for x in range(0, 10)]:
        lambda_x = x
        win = 0
        for t in range(runs):
            #initialize our records
            results = []
            # start the game
            env.start()
            # initialize first state and action
            action = ep_greedy(env.state())
            
            # initialize our eligibility traces for each episode
            traces = np.zeros((DEALERS_FEATURE * PLAYERS_FEATURE * ACTIONS_FEATURE), dtype=np.float32)
            # initialize weights should have the same number as features
            weights = np.random.uniform(0,1, size=DEALERS_FEATURE * PLAYERS_FEATURE * ACTIONS_FEATURE)
            q_states_actions = np.zeros((DEALERS_STATE, PLAYERS_STATE, ACTIONS_FEATURE), dtype=np.float32)

            while not env.is_finished():
                current_state = env.state()
                current_dealer, current_player = current_state
                # take action A, observe S', Reward
                next_state, reward = env.step(current_state, action)
                next_dealer, next_player = next_state
                # choose an action based on policy with features (http://artint.info/html/ArtInt_272.html)
                current_q_values = Q_approx(current_dealer-1, current_player-1, action, weights)
                current_feature_vec = Features(current_dealer-1, current_player-1, action)
                
                # if we should still proceed
                if(next_dealer <= 10 and next_player <= 21):
                    # choose A' from S' using e-greedy
                    next_action = ep_greedy(next_state)

                    # calculate the TD error (delta)
                    next_q_values = Q_approx(current_dealer-1, current_player-1, action, weights)
                    td_error = reward + (GAMMA * next_q_values) - current_q_values
                else:
                    td_error = reward - current_q_values

                # backward view lambda
                traces = GAMMA * lambda_x * traces + current_feature_vec
                weights += ALPHA * td_error * traces
                if reward == 1:
                    win += 1
            
            

            # Read the q true from disk
            q_true_txt = np.loadtxt('q_true.txt')

            # original shape of the array
            q_true = q_true_txt.reshape((DEALERS_STATE,PLAYERS_STATE,ACTIONS_FEATURE))
            q_diff = q_states_actions - q_true
            #print(np.mean(np.square(q_diff)))

        print ("lambda %f, won %d" % (lambda_x, win))
if __name__ == '__main__':
    main()