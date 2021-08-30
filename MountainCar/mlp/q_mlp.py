import os
import math
import time
import numpy as np
import random
import pandas as pd
from collections import deque
from logger.score import ScoreLogger
import gym

import tensorflow as tf
from tensorflow import keras

import csv
from time import strftime
import neptune


neptune.init(project_qualified_name='v3rm1/MC-QRTM')

seed_values = [2, 131, 1729]#, 4027, 10069]

# Reward decay
GAMMA = 0.99
# Learning Rate
ALPHA = 0.01

# Experience-Replay Memory Parameters
MEMORY_SIZE = 10000
BATCH_SIZE = 200

# Exploration-Exploitation Parameters
EPSILON_MIN = 0.01
EPSILON_MAX = 1
EPSILON_DECAY = 0.99

# Number Of Episodes to run
EPISODES = 3000

EXPT_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "expt_csv/expts_"+strftime("%Y%m%d_%H%M%S")+".csv")



class DQNAgent:

    def __init__(self, environment):
        super().__init__()
        self.obs_space = environment.observation_space.shape[0]
        self.action_space = environment.action_space.n

        self.memory = deque(maxlen=MEMORY_SIZE)
        # self.memory = []

        self.epsilon = EPSILON_MAX
        self.q_net = self.network()

    def network(self):

        self.model = keras.Sequential()
        self.model.add(
            keras.layers.Dense(400,
                               input_shape=(self.obs_space, ),
                               activation="relu"))
        self.model.add(
            keras.layers.Dense(300,
                               activation="relu"))
        self.model.add(
            keras.layers.Dense(self.action_space, activation="linear"))
        self.model.compile(
            loss="mse", optimizer=keras.optimizers.Adam(learning_rate=ALPHA))
        return self.model

    def memorize(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.q_net.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self, episode, log_file):

        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        # batch = self.memory[-BATCH_SIZE:]
        for state, action, reward, next_state, done in batch:
            q_update = reward
            print("q_update before discount:{}".format(q_update), file=open(log_file, 'a'))
            if not done:
                q_update = reward + GAMMA * np.amax(self.q_net.predict(next_state)[0])
                print("q_update after discount:{}".format(q_update), file=open(log_file, 'a'))
            q_values = self.q_net.predict(state)
            q_values[0][action] = q_update
            # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)
            self.q_net.fit(state, q_values, verbose=0)
        self.epsilon = EPSILON_MAX * pow(EPSILON_DECAY, episode)
        self.epsilon = max(EPSILON_MIN, self.epsilon)


def main():
    expt_data = pd.DataFrame()
    for seed_value in seed_values:
        
        # NOTE: DEFINING A STDOUT LOGGER and BINARIZER DISTRIBUTION PLOT PATHS
        STDOUT_LOG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logger/txt_logs/run_"+strftime("%Y%m%d_%H%M%S")+"_"+str(seed_value)+".txt")
        # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
        os.environ['PYTHONHASHSEED']=str(seed_value)
        # 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed_value)
        # 3. Set `numpy` pseudo-random generator at a fixed value
        np.random.seed(seed_value)

        env = gym.make("MountainCar-v0")
        score_log = ScoreLogger("MountainCar-v0")
        neptune.create_experiment(name="MLP", tags=["MLP"])

        dqn_agent = DQNAgent(env)
        for ep in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, env.observation_space.shape[0]])

            accum_reward = 0
            episode_len = 0
            done = False
            while not done:
                episode_len += 1
                action = dqn_agent.act(state)
                next_state, reward, done, info = env.step(action)

                next_state = np.reshape(next_state,
                                        [1, env.observation_space.shape[0]])
                accum_reward += reward
                state = next_state
                if done and episode_len < 200:
                    # Reward egineering: if goal is reached in less than 200 steps, reward = episode reward + 250
                    reward = 100
                    accum_reward += reward
                    print("Episode: {0}\nEpsilon: {1}\tScore: {2}".format(
                        ep, dqn_agent.epsilon, accum_reward), file=open(STDOUT_LOG, 'a'))
                    score_log.add_score(episode_len, ep)
                elif done:
                    accum_reward += reward
                    print("Episode: {0}\nEpsilon: {1}\tScore: {2}".format(
                        ep, dqn_agent.epsilon, accum_reward), file=open(STDOUT_LOG, 'a'))
                    score_log.add_score(episode_len, ep)
                    # reward engineering for other steps: reward = distance travelled + velocity
                    print("Reward: {}".format(reward), file=open(STDOUT_LOG, 'a'))
                dqn_agent.memorize(state, action, reward, next_state, done)
            dqn_agent.experience_replay(ep, STDOUT_LOG)
            
            neptune.log_metric('steps', episode_len)
            neptune.log_metric('accum_reward', accum_reward)
        # Add experiment columns to the dataframe
        expt_data.loc[:, 'score_'+str(seed_value)] = episode_len
    expt_data.to_csv(EXPT_DATA)


if __name__ == "__main__":
    main()
