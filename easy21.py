import numpy as np

class Easy21:
    def __init__(self):
        self.dealers_hand = 0
        self.players_sum = 0
        self.finished = False
    
    def draw(self):
        if self.finished == True:
            print ("game is already finished")
            assert RuntimeError()

        card_value = int(np.random.uniform(1, 11))
        # probabilistically generate red 0 or black 1
        card_color = np.random.choice(np.arange(0, 2), p=[0.335, 0.665])
        return card_color, card_value

    def start(self):
        self.finished = False
        self.dealers_hand = 0
        self.players_sum = 0
        card_color, card_value = self.draw()
        # we assume we always get the black at the start
        # so no need to check card_color
        self.dealers_hand += card_value
        card_color, card_value = self.draw()
        self.players_sum += card_value
        

    def state(self):
        return (self.dealers_hand, self.players_sum)

    def game_finished(self):
        print ("setting game to finished state")
        self.finished = True

    def is_finished(self):
        return self.finished

    def step(self, state, action):
        """
            params: 
                state - (dealers, player)  in the current state of the game
                action - int 0 indicate stick, 1 otherwise
            return:
                next_state - (dealers, player)
                reward - 
                    -1, lost
                    0, drew
                    1, win
        """
        if self.finished == True:
            print ("game is already finished")
            assert RuntimeError()

        reward = 0
        dealer, player = state
        # if we stick, return the sample next state
        # and the reward, depends on whether dealer's win or lose.
        if action == 0:
            # dealer's turn
            while self.dealers_hand < 17:
                if self.dealers_hand < 1:
                    self.game_finished()
                    return self.state(), reward

                card_color, card_value = self.draw()
                if card_color == 0:
                    self.dealers_hand -= card_value
                else:
                    self.dealers_hand += card_value
            
            # dealer is greater than 17 and he bust :)!
            if self.dealers_hand > 21:
                reward += 1
                self.game_finished()
                return self.state(), reward
            
            # dealer hand is greater than 17 but not bust :(
            if (self.players_sum - self.dealers_hand) > 0:
                # we win
                reward += 1
                self.game_finished()
                return self.state(), reward
            elif self.players_sum - self.dealers_hand == 0:
                # draw :/
                self.game_finished()
                return self.state(), reward
            else:
                # lost
                reward -= 1
                self.game_finished()
                return self.state(), reward
    
        # we chose hit
        card_color, card_value = self.draw()
        if card_color == 0:
            self.players_sum -= card_value
        else:
            self.players_sum += card_value

        # would we bust ?
        if self.players_sum > 21:
            reward -= 1
            self.game_finished()
        
        # we don't need any state that is below 1
        if self.players_sum < 1:
            self.game_finished()
        
        return self.state(), reward