from os import path
import yaml
import numpy as np
import random
import math
from collections import deque
from time import strftime
import csv
import gym
from logger.score import ScoreLogger
from discretizer import CustomDiscretizer
from debug_plot_functions import DebugLogger
from rtm import TsetlinMachine

# Path to file containing all configurations for the variables used by the q-rtm system
CONFIG_PATH = path.join(path.dirname(path.realpath(__file__)), 'config.yaml')
#
CONFIG_TEST_SAVE_PATH = path.join(path.dirname(path.realpath(__file__)), 'tested_configs.csv')

# NOTE: DEFINING A STDOUT LOGGER TO STORE ALL PRINT STATEMENTS FOR FUTURE USE
STDOUT_LOG = path.join(path.dirname(path.realpath(__file__)), "run_"+strftime("%Y%m%d_%H%M%S")+".txt")

class RTMQL:
    def __init__(self, environment, config, epsilon_decay_config="EDF"):
        super().__init__()

        # Environment config
        self.obs_space = environment.observation_space.shape[0]
        self.action_space = environment.action_space.n

        self.memory = deque(maxlen=config['memory_params']['memory_size'])
        self.replay_batch = config['memory_params']['batch_size']

        self.episodes = config['game_params']['episodes']
        self.reward = config['game_params']['reward']
        self.max_score = config['game_params']['max_score']
        self.min_score = config['game_params']['min_score']

        self.gamma = config['learning_params']['gamma']
        self.learning_rate = config['learning_params']['learning_rate']
        
        self.weighted_clauses = config['qrtm_params']['weighted_clauses']
        self.incremental = config['qrtm_params']['incremental']
        self.ta_states = config['qrtm_params']['ta_states']

        
        self.epsilon_max = config['learning_params']['EDF']['epsilon_max']
        self.eps_decay = epsilon_decay_config
        self.epsilon_min = config['learning_params']['EDF']['epsilon_min']

        self.epsilon = self.epsilon_max

        self.T = config['qrtm_params']['T']
        self.s = config['qrtm_params']['s']
        self.number_of_clauses = config['qrtm_params']['number_of_clauses']
        self.number_of_features = config['qrtm_params']['number_of_features']

        if epsilon_decay_config == "SEDF":
            self.sedf_alpha = config['learning_params']['SEDF']['tail']
            self.sedf_beta = config['learning_params']['SEDF']['slope']
            self.sedf_delta = config['learning_params']['SEDF']['tail_gradient']
            print("Agent configured to use Stretched Exponential Decay Function for Epsilon value.\nAlpha (tail): {}\nBeta (slope): {}\nDelta (tail_gradient): {}".format(self.sedf_alpha, self.sedf_beta, self.sedf_delta))
        else:
            self.epsilon_min = config['learning_params']['EDF']['epsilon_min']
            self.epsilon_max = config['learning_params']['EDF']['epsilon_max']
            self.epsilon_decay = config['learning_params']['EDF']['epsilon_decay']
            print("Agent configured to use Exponential Decay Function for Epsilon value.\nDecay: {}\nMax Epsilon: {}\nMin Epsilon: {}".format(self.epsilon_decay, self.epsilon_max, self.epsilon_min))

        self.agent_1 = self.tm_model()
        self.agent_2 = self.tm_model()

    def exp_eps_decay(self, current_ep):
        self.epsilon = self.epsilon_max * pow(self.epsilon_decay, current_ep)
        return max(self.epsilon_min, self.epsilon)
    
    def stretched_exp_eps_decay(self, current_ep):
        self.epsilon = 1.1 - (1 / (np.cosh(math.exp(-(current_ep - self.sedf_alpha * self.episodes) / (self.sedf_beta * self.episodes)))) + (current_ep * self.sedf_delta / self.episodes))
        return max(min(self.epsilon_max, self.epsilon), self.epsilon_min)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def tm_model(self):
        self.tm_agent = TsetlinMachine(
            number_of_clauses = self.number_of_clauses,
            number_of_features = self.number_of_features,
            number_of_states=self.ta_states,
            s=self.s,
            threshold=self.T,
            max_target=self.max_score,
            min_target=self.min_score,
            logger=STDOUT_LOG
        )
        return self.tm_agent

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            a = random.randrange(self.action_space)
            return a
        q_values = [self.agent_1.predict(state), self.agent_2.predict(state)]
        return np.argmax(q_values)
    
    def experience_replay(self, episode):
        if len(self.memory) < self.replay_batch:
            return [0,0]
        # Generate random batch from memory
        batch = random.sample(self.memory, self.replay_batch)

        for state, action, reward, next_state, done in batch:
            q_update = 0
            # Compute q-values for state
            q_values = [self.agent_1.predict(state), self.agent_2.predict(state)]
            # Compute extected q-values for next state
            target_q = [self.agent_1.predict(next_state), self.agent_2.predict(next_state)]

            print("PRE FIT", file=open(STDOUT_LOG, "a"))
            print("Q_Values - State: {}".format(q_values), file=open(STDOUT_LOG, "a"))
            print("Expectation - Next State: {}".format(target_q), file=open(STDOUT_LOG, "a"))

            # Compute temporal difference error
            td_error = reward + self.gamma * np.amax(target_q) - q_values[action]

            if done:
                # If game is done, the update equals reward
                q_update = reward
            else:
                # If game is not done, compute update using temporal difference error
                q_update += self.learning_rate * td_error
            
            # Add update to q_value of action taken for state
            q_values[action] += q_update

			# Update agents on new q-values for the state
            self.agent_1.update(state, q_values[action])
            self.agent_2.update(state, q_values[1-action])

            # td_err_post_fit = reward + self.gamma * np.amax([self.agent_1.predict(next_state), self.agent_2.predict(next_state)]) - q_values[action]

            # print("TD_ERROR Pre fit: {}\nTD_ERROR Post fit: {}".format(td_error, td_err_post_fit))
            print("POST FIT", file=open(STDOUT_LOG, "a"))
            print("Q_Values - State: {}".format([self.agent_1.predict(state), self.agent_2.predict(state)]), file=open(STDOUT_LOG, "a"))
            print("Expectation - Next State: {}".format([self.agent_1.predict(next_state), self.agent_2.predict(next_state)]), file=open(STDOUT_LOG, "a"))



        # Epsilon decay
        if self.eps_decay == "SEDF":
            self.epsilon = self.stretched_exp_eps_decay(episode)
        else:
            # Exponential epsilon decay
            self.epsilon = self.exp_eps_decay(episode)
        return td_error**2
        
def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def store_config_tested(config_data, win_count, run_date, tested_configs_file_path=CONFIG_TEST_SAVE_PATH):
    # Defining dictionary key mappings
    field_names = ['decay_fn', 'epsilon_min', 'epsilon_max', 'epsilon_decay', 'alpha', 'beta', 'delta', 'reward_discount', 'mem_size', 'batch_size', 'episodes', 'reward', 'max_score', 'min_score', 'action_space', 'qrtm_n_clauses', 'ta_states', 'T', 's', 'wins', 'win_ratio', 'run_date', 'bin_length', 'incremental', 'weighted_clauses', 'binarizer']
    decay_fn = config_data['learning_params']['epsilon_decay_function']
    if decay_fn == "SEDF":
        alpha = config_data['learning_params']['SEDF']['tail']
        beta = config_data['learning_params']['SEDF']['slope']
        delta = config_data['learning_params']['SEDF']['tail_gradient']
        eps_min = 0
        eps_max = 0
        eps_decay = 0
    else:
        alpha = 0
        beta = 0
        delta = 0
        eps_min = config_data['learning_params']['EDF']['epsilon_min']
        eps_max = config_data['learning_params']['EDF']['epsilon_max']
        eps_decay = config_data['learning_params']['EDF']['epsilon_decay']
    store_config = {
        'decay_fn': decay_fn,
        'epsilon_min': eps_min,
        'epsilon_max': eps_max,
        'epsilon_decay': eps_decay,
        'alpha': alpha,
        'beta': beta,
        'delta': delta,
        'reward_discount': config_data['learning_params']['gamma'],
        'mem_size': config_data['memory_params']['memory_size'],
        'batch_size': config_data['memory_params']['batch_size'],
        'episodes': config_data['game_params']['episodes'],
        'reward': config_data['game_params']['reward'],
        'max_score': config_data['game_params']['max_score'],
        'min_score': config_data['game_params']['min_score'],
        'action_space': config_data['game_params']['action_space'],
        'qrtm_n_clauses': config_data['qrtm_params']['number_of_clauses'],
        'ta_states': config_data['qrtm_params']['ta_states'],
        'T': config_data['qrtm_params']['T'],
        's': config_data['qrtm_params']['s'],
        'wins': win_count,
        'win_ratio': win_count/config_data['game_params']['episodes'],
        'run_date': run_date,
        'bin_length':config_data['qrtm_params']['feature_length'],
        'incremental':config_data['qrtm_params']['incremental'],
        'weighted_clauses': config_data['qrtm_params']['weighted_clauses'],
        'binarizer': config_data['preproc_params']['binarizer']
    }
    # Write to file. Mode a creates file if it does not exist.
    if not path.exists(tested_configs_file_path):
        with open(tested_configs_file_path, 'w', newline='') as write_obj:
            header_writer = csv.writer(write_obj)
            header_writer.writerow(field_names)
    with open(tested_configs_file_path, 'a+', newline='') as write_obj:
        dict_writer = csv.DictWriter(write_obj, fieldnames=field_names)
        dict_writer.writerow(store_config)
    return

def main():
    config = load_config(CONFIG_PATH)
    gamma = config['learning_params']['gamma']
    episodes = config['game_params']['episodes']
    run_dt = strftime("%Y%m%d_%H%M%S")
    epsilon_decay_function = config['learning_params']['epsilon_decay_function']
    feature_length = config['qrtm_params']['feature_length']
    print("Configuration file loaded. Creating environment.", file=open(STDOUT_LOG, "a"))
    env = gym.make("CartPole-v0")
    
    # Initializing loggers and watchers
    debug_log = DebugLogger("CartPole-v0")
    score_log = ScoreLogger("CartPole-v0", episodes)

    print("Initializing custom discretizer.", file=open(STDOUT_LOG, "a"))
    discretizer = CustomDiscretizer()
    print("Initializing Q-RTM Agent.", file=open(STDOUT_LOG, "a"))
    rtm_agent = RTMQL(env, config, epsilon_decay_function)
    binarized_length = int(config['qrtm_params']['feature_length'])
    binarizer = config['preproc_params']['binarizer']

    # Initializing experiment variables and data structures
    td_error = []
    win_ctr = 0

    for curr_ep in range(episodes):
        
        td_err_ep = []
        # Initialize episode variables
        step = 0
        done = False

        # Reset state to start
        state = env.reset()
        # Discretize and reshape state
        state = discretizer.cartpole_binarizer(input_state=state, n_bins=binarized_length-1, bin_type=binarizer)
        state = np.reshape(state, [1, feature_length * env.observation_space.shape[0]])[0]

        while not done:
            step += 1
            # Request agent action
            action = rtm_agent.act(state)
            
            # Run simulation step in environment, retrieve next state, reward and game status
            next_state, reward, done, info = env.step(action)
            
            # Discretize and reshape next_state
            next_state = discretizer.cartpole_binarizer(input_state=next_state, n_bins=binarized_length-1, bin_type=binarizer)
            next_state = np.reshape(next_state, [1, feature_length * env.observation_space.shape[0]])[0]

            # Memorization
            rtm_agent.memorize(state, action, reward, next_state, done)

            # Set state to next state
            state = next_state

            # Game end condition
            if done:
                # Increment win counter conditionally
                if step > 195:
                    win_ctr += 1

                print("Episode: {0}\nEpsilon: {1}\tScore: {2}".format(curr_ep, rtm_agent.epsilon, step), file=open(STDOUT_LOG, "a"))
                score_log.add_score(step,
                curr_ep,
                gamma,
                epsilon_decay_function,
                consecutive_runs=episodes,
                sedf_alpha=config['learning_params']['SEDF']['tail'],
                sedf_beta=config['learning_params']['SEDF']['slope'],
                sedf_delta=config['learning_params']['SEDF']['tail_gradient'],
                edf_epsilon_decay=config['learning_params']['EDF']['epsilon_decay'])
                break
        
        # Store TD error from experience replay
        td_err_ep.append(rtm_agent.experience_replay(curr_ep))
        # Append average TD error per episode to list
        td_error.append(np.sqrt(np.mean(td_err_ep)))
    print("Len of TDERR array: {}".format(len(td_error)))
    # Plot average TD error over episode
    debug_log.add_watcher(td_error,
                          n_clauses=config["qrtm_params"]["number_of_clauses"],
                          T=config["qrtm_params"]["T"],
                          feature_length=feature_length)

    # Print win counter
    print("win_ctr: {}".format(win_ctr), file=open(STDOUT_LOG, "a"))

    # Store configuration tested, win count and timestamp of experiment
    store_config_tested(config, win_ctr, run_dt)


if __name__ == "__main__":
    main()
                

