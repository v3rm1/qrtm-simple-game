import numpy as np

STEP_REWARD = 1
class Cartpole_Simplified:
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.end_states = [0, self.n_states-1]
        self.observation_space = np.arange(0, self.n_states, 1)
        # Action space: 0 = left, 1 = right
        self.action_space = [0, 1]
        self.start_state = self._sample_state()
        self.current_state = self.start_state
        self.reward = STEP_REWARD
        self.done = False

    def _sample_state(self):
        entry_states = np.setdiff1d(self.observation_space, self.end_states)
        return np.random.choice(entry_states)

    def _check_terminal(self, state):
        if state in self.end_states:
            if state == self.end_states[0]:
                self.reward = -1
                self.done = True
            else:
                self.reward = 1
                self.done = True
        return self.reward, self.done
    
    def _act_step(self, action):
        current_state = self.current_state
        if action in self.action_space:
            if action == 0:
                next_state = current_state - 1
                # self.reward += 1
            else:
                next_state = current_state + 1
                # self.reward += 1
        return current_state, next_state
    
    def _update_state(self, nxt_state):
        self.current_state = nxt_state
        
    
    def game_step(self, action):
        curr, nxt = self._act_step(action)
        rew, done = self._check_terminal(nxt)
        self._update_state(nxt)
        return curr, nxt, rew, done

    def reset(self):
        self.end_states = [0, self.n_states-1]
        self.observation_space = [np.arange(0, self.n_states, 1)]
        # Action space: 0 = left, 1 = right
        self.action_space = [0, 1]
        self.start_state = self._sample_state()
        self.current_state = self.start_state
        self.reward = STEP_REWARD
        self.done = False
        return self.current_state

def test_game():
    sim_cp = Cartpole_Simplified()
    print("Start state: {}".format(sim_cp.start_state))
    done = False
    while not done:
        a = int(input("Enter action [0/1]:"))
        curr_st, next_st, r, done = sim_cp.game_step(a)
        print("curr_st: {0}\nnext_st: {1}\nreward: {2}\naction: {3}".format(curr_st, next_st, r, a))
    return


if __name__ == "__main__":
    test_game()
