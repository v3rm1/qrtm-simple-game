import os
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
from logger.debugger import DebugLogger
from rtm import TsetlinMachine
from per_memory import Memory
import sys

import neptune


neptune.init(project_qualified_name='v3rm1/MC-QRTM')

# NOTE: SETTING GLOBAL SEED VALUES FOR CONSISTENT RESULTS IN EXPERIMENTAL SESSIONS
# Set a seed value
seed_value = 4027
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# A variable for attaching test tag to the experiment
TEST_VAR = True

# Path to file containing all configurations for the variables used by the q-rtm system
CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yaml')
#
CONFIG_TEST_SAVE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tested_configs.csv')

# NOTE: DEFINING A STDOUT LOGGER TO STORE ALL PRINT STATEMENTS FOR FUTURE USE
STDOUT_LOG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logger/txt_logs/run_"+strftime("%Y%m%d_%H%M%S")+".txt")

BIN_DIST_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logger/bin_dist/bin_dist"+strftime("%Y%m%d_%H%M%S")+".png")

class RTMQL:
	def __init__(self, environment, config, epsilon_decay_config="EDF"):
		super().__init__()

		# Environment config
		self.obs_space = environment.observation_space.shape[0]
		self.action_space = environment.action_space.n

		self.memory = Memory(config['memory_params']['memory_size'])
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
		q_vals = [self.agent_1.predict(state), self.agent_2.predict(state)]
		target_q = [self.agent_1.predict(next_state), self.agent_2.predict(next_state)]

		q_action = q_vals[action]

		if done:
			q_update = reward
		else:
			q_update = self.learning_rate * (reward + self.gamma * target_q[action]-q_vals[action])
		q_vals[action] += q_update

		error = abs(q_action - target_q[action])
		self.memory.add_sample_to_tree(error, (state, action, reward, next_state, done))
		return error

	def tm_model(self):
		self.tm_agent = TsetlinMachine(
			number_of_clauses = self.number_of_clauses,
			number_of_features = self.number_of_features,
			number_of_states=self.ta_states,
			s=self.s,
			threshold=self.T,
			max_target=self.max_score,
			min_target=self.min_score
		)
		return self.tm_agent

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			print("RANDOM ACTION")
			a = random.randrange(self.action_space)
			return a
		print("LEARNER ACTION")
		q_values = [self.agent_1.predict(state), self.agent_2.predict(state)]
		return np.argmax(q_values)
	
	def experience_replay(self, episode):
		td_err_list = []
		q_val_list = []
		batch, idxs, is_weights = self.memory.sample_tree(self.replay_batch)
		batch = np.array(batch, dtype=object).transpose()

		states = np.vstack(batch[0])
		actions = list(batch[1])
		rewards = list(batch[2])
		next_states = np.vstack(batch[3])
		done_list = list(batch[4])

		for idx, state, action, reward, next_state, done in zip(idxs, states, actions, rewards, next_states, done_list):
			q_values = [self.agent_1.predict(state), self.agent_2.predict(state)]
			target_q = [self.agent_1.predict(next_state), self.agent_2.predict(next_state)]
			td_error = reward + self.gamma * np.amax(target_q) - q_values[action]

			if done:
				q_update = reward
			else:
				q_update = self.learning_rate * td_error
			
			q_values[action] += q_update

			target = reward * (1 - done) * self.gamma * target_q[action]

			error = abs(q_values[action] - target)

			self.memory.update_tree(idx, error)

			self.agent_1.update(state, q_values[0])
			self.agent_2.update(state, q_values[1])

			print("TD_ERROR: {}".format(td_error))
			td_err_list.append(pow(td_error, 2))
			q_val_list.append(q_values)
		
		rms_td_err = np.sqrt(np.mean(td_err_list))
		qmax_init = np.max(q_val_list[0])

		# Epsilon decay
		if self.eps_decay == "SEDF":
			self.epsilon = self.stretched_exp_eps_decay(episode)
		else:
			# Exponential epsilon decay
			self.epsilon = self.exp_eps_decay(episode)
		return rms_td_err, qmax_init
		
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
	if not os.path.exists(tested_configs_file_path):
		with open(tested_configs_file_path, 'w', newline='') as write_obj:
			header_writer = csv.writer(write_obj)
			header_writer.writerow(field_names)
	with open(tested_configs_file_path, 'a+', newline='') as write_obj:
		dict_writer = csv.DictWriter(write_obj, fieldnames=field_names)
		dict_writer.writerow(store_config)
	return

def main():
	neptune.create_experiment(name="RTM-PER", tags=["local"])

	if TEST_VAR:
		neptune.append_tag("test")

	config = load_config(CONFIG_PATH)
	gamma = config['learning_params']['gamma']
	episodes = config['game_params']['episodes']
	run_dt = strftime("%Y%m%d_%H%M%S")
	epsilon_decay_function = config['learning_params']['epsilon_decay_function']
	feature_length = config['qrtm_params']['feature_length']
	print("Configuration file loaded. Creating environment.")
	env = gym.make("MountainCar-v0")
	if gamma<1:
		neptune.append_tag("gamma="+str(gamma))


	neptune.log_text('T', str(config['qrtm_params']['T']))
	neptune.log_text('s', str(config['qrtm_params']['s']))
	neptune.log_text('Feature length (bits/feature)', str(config['qrtm_params']['feature_length']))
	neptune.log_text('Number of Clauses', str(config['qrtm_params']['number_of_clauses']))
	neptune.log_text('Number of TA States', str(config['qrtm_params']['ta_states']))
	neptune.log_text('Binarizer', str(config['preproc_params']['binarizer']))
	neptune.log_text('Exp Replay Batch', str(config['memory_params']['batch_size']))
	neptune.log_text('Epsilon Decay Function', str(config['learning_params']['epsilon_decay_function']))
	neptune.log_text('Gamma', str(config['learning_params']['gamma']))
	
	# Initializing loggers and watchers
	debug_log = DebugLogger("MountainCar-v0")
	score_log = ScoreLogger("MountainCar-v0", episodes)

	print("Initializing custom discretizer.")
	discretizer = CustomDiscretizer()
	print("Initializing Q-RTM Agent.")
	rtm_agent = RTMQL(env, config, epsilon_decay_function)
	binarized_length = int(config['qrtm_params']['feature_length'])
	binarizer = config['preproc_params']['binarizer']

	# Initializing experiment variables and data structures
	td_error = []
	win_ctr = 0

	for curr_ep in range(episodes):
		
		rms_td_err_ep = 0
		# Initialize episode variables
		step = 0
		done = False

		# Reset state to start
		state = env.reset()
		# Discretize and reshape state
		state = discretizer.cartpole_binarizer(input_state=state, n_bins=binarized_length, bin_type=binarizer)
		state = np.reshape(state, [1, feature_length * env.observation_space.shape[0]])[0]
		

		while not done:
			step += 1
			# Request agent action
			action = rtm_agent.act(state)
			print("State: {0}".format(state))
			# Run simulation step in environment, retrieve next state, reward and game status
			next_state, reward, done, info = env.step(action)
			reward = reward if not done else -reward
			# Discretize and reshape next_state
			next_state = discretizer.cartpole_binarizer(input_state=next_state, n_bins=binarized_length, bin_type=binarizer)
			next_state = np.reshape(next_state, [1, feature_length * env.observation_space.shape[0]])[0]
			print("Next State: {0}".format(next_state))
			# Memorization
			rtm_agent.memorize(state, action, reward, next_state, done)

			# Set state to next state
			state = next_state

			# Game end condition
			if done:
				# Increment win counter conditionally
				if step > 195:
					win_ctr += 1

				print("Episode: {0}\nEpsilon: {1}\tScore: {2}".format(curr_ep, rtm_agent.epsilon, step))
				score_log.add_score(step,
				curr_ep,
				gamma,
				epsilon_decay_function,
				consecutive_runs=episodes,
				sedf_alpha=config['learning_params']['SEDF']['tail'],
				sedf_beta=config['learning_params']['SEDF']['slope'],
				sedf_delta=config['learning_params']['SEDF']['tail_gradient'],
				edf_epsilon_decay=config['learning_params']['EDF']['epsilon_decay'])
				neptune.log_metric('score', step)
				break
		# if step < 195:	
		# 	# Store TD error from experience replay
		rms_td_err_ep, qmax_init = rtm_agent.experience_replay(curr_ep)
		# else:
		# 	rms_td_err_ep = 0
		print("episode td err RMS: {}".format(rms_td_err_ep))
		# Append average TD error per episode to list
		td_error.append(rms_td_err_ep)
		neptune.log_metric('TD_ERR (RMS)', rms_td_err_ep)
		neptune.log_metric('Max Init Q', qmax_init)
		
	print("Len of TDERR array: {}".format(len(td_error)))

	# Plot average TD error over episode
	debug_log.add_watcher(td_error,
						  n_clauses=config["qrtm_params"]["number_of_clauses"],
						  T=config["qrtm_params"]["T"],
						  feature_length=feature_length)
	
	
	# Print win counter
	print("win_ctr: {}".format(win_ctr))

	# Store configuration tested, win count and timestamp of experiment
	store_config_tested(config, win_ctr, run_dt)

	neptune.log_artifact(CONFIG_PATH)
	
	discretizer.plot_bin_dist(plot_file=BIN_DIST_FILE, binarizer=binarizer)
	


if __name__ == "__main__":
	sys.stdout = open(STDOUT_LOG, 'w')
	main()
	sys.stdout.close()
