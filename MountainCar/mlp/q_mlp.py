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


# Reward decay
GAMMA = 0.99
# Learning Rate
ALPHA = 0.001

# Experience-Replay Memory Parameters
MEMORY_SIZE = 10000
BATCH_SIZE = 64

# Exploration-Exploitation Parameters
EPSILON_MIN = 0.01
EPSILON_MAX = 1
EPSILON_DECAY = 0.99

# Number Of Episodes to run
EPISODES = 5000

STDOUT_LOG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "run_"+strftime("%Y%m%d_%H%M%S")+".txt")

class DQNAgent:
    """ """
    def __init__(self, environment):
        super().__init__()
        self.obs_space = environment.observation_space.shape[0]
        self.action_space = environment.action_space.n

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.epsilon = EPSILON_MAX
        self.q_net = self.network()

    def network(self):
        """ """
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
        """

        :param state: 
        :param action: 
        :param reward: 
        :param next_state: 
        :param done: 

        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """

        :param state: 

        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.q_net.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self, episode):
        """ """
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            print("q_update before discount:{}".format(q_update), file=open(STDOUT_LOG, 'a'))
            if not done:
                q_update = reward + GAMMA * np.amax(self.q_net.predict(next_state)[0])
                print("q_update after discount:{}".format(q_update), file=open(STDOUT_LOG, 'a'))
            q_values = self.q_net.predict(state)
            q_values[0][action] = q_update
            # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)
            self.q_net.fit(state, q_values, verbose=0)
        self.epsilon = EPSILON_MAX * pow(EPSILON_DECAY, episode)
        self.epsilon = max(EPSILON_MIN, self.epsilon)


def main():
    """ """
    env = gym.make("MountainCar-v0")
    score_log = ScoreLogger("MountainCar-v0")
    neptune.create_experiment(name="MLP", tags=["peregrine"])

    dqn_agent = DQNAgent(env)
    for ep in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        episode_reward = 0
        episode_len = 0
        done = False
        while not done:
            episode_len += 1
            action = dqn_agent.act(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            next_state = np.reshape(next_state,
                                    [1, env.observation_space.shape[0]])
            
            state = next_state
            if done and episode_len < 200:
                # Reward egineering: if goal is reached in less than 200 steps, reward = episode reward + 250
                reward = 100
                print("Episode: {0}\nEpsilon: {1}\tScore: {2}".format(
                    ep, dqn_agent.epsilon, reward), file=open(STDOUT_LOG, 'a'))
                score_log.add_score(episode_len, ep)
            elif done:
                print("Episode: {0}\nEpsilon: {1}\tScore: {2}".format(
                    ep, dqn_agent.epsilon, episode_reward), file=open(STDOUT_LOG, 'a'))
                score_log.add_score(episode_len, ep)
                # reward engineering for other steps: reward = distance travelled + velocity
                print("Reward: {}".format(reward), file=open(STDOUT_LOG, 'a'))
            dqn_agent.memorize(state, action, reward, next_state, done)
        dqn_agent.experience_replay(ep)
        
        neptune.log_metric('steps', episode_len)
        neptune.log_metric('accum_reward', episode_reward)
        neptune.log_metric('manip_reward', reward)


if __name__ == "__main__":
    main()
