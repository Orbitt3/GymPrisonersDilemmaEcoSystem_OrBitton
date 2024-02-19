# class of learner which can change agents rewards and wants to maximize cooperation
import numpy as np
import random
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
import matplotlib
from IPython import display
from tqdm import tqdm
from agent import Agent
import tensorflow as tf
from PdeMetaAgentEnv import PdeMetaAgentEnv
from PdeSingleAgentEnv import PdeSingleAgentEnv
from strategyagent import StrategyAgent
from collections import namedtuple, deque
import collections
import itertools
from itertools import count
import math
from MetaAgent import MetaAgent
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dqnAgent import DQNAgent
from torch.distributions import Categorical
import logging
import warnings
from MetaAgentClass import MetaDQNAgent

# Filter tensorflow version warnings
import os
# # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=Warning)
# tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(0)
# tf.get_logger().setLevel(logging.ERROR)

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class PdeVecEnv(gym.Env):
    # init env, set memory size, game rewards, history length
    def __init__(self, memory_size=2):
        self.env_name = 'Meta_PDE'

        # low level params:
        self.memory_size = memory_size
        self.low_env = PdeSingleAgentEnv(memory_size)
        self.low_level_episode_length = 1000

        # high level params:
        self.change_options = 3
        self.n_actions = int(self.low_env.score.reshape(8, -1).shape[0])*self.change_options  # 8X1[0]
        self.n_observations = self.low_env.score.reshape(8, -1).shape[0]  # 8X1[0]
        self.change_factor = 2

        self.history = []
        self.meta_episode_len = 8
        self.cell_list = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
        self.bounds = [-9, 9]
        self.reward_memory = np.zeros(int(10e6))

        self.games_counter = 0
        self.done = False

    def get_reward(self):
        # count rewards for meta agent:
        meta_hist = self.low_env.last100_history  # deque of 100 length 2 lists
        meta_hist = [item for sublist in meta_hist for item in sublist if sublist[0] == 1 and sublist[1] == 1]  # flatten list
        meta_reward = meta_hist.count(1)  # count number of co-op actions in game hist

        # meta_reward = self.low_env.score[3, 1] + self.low_env.score[3, 0]  # get meta reward from score table
        return meta_reward

    def check_valid_bounds(self, change, cell):
        new_val = self.low_env.score[cell[0], cell[1]] + change
        if new_val <= self.bounds[1] and new_val >= self.bounds[0]:
            return True
        return False

    def update_score_table(self, change, cell):
        if self.check_valid_bounds(change, cell):
            self.low_env.score[cell[0], cell[1]] += change

    def action_to_change_cell(self, selection):
        """
        get selection from 0 to 23(num_actions) and returns action and cell
        """
        change = selection % self.change_options  # 0,1,2
        cell = int(selection / self.change_options)  # 0,...,7

        change = self.map_argmax_change_to_change(change)
        cell = self.map_argmax_cell_to_cell(cell)
        return change, cell

    def map_argmax_change_to_change(self, change):
        modified_change = change - 1
        modified_change = modified_change * self.change_factor
        return modified_change

    def map_argmax_cell_to_cell(self, cell):  # cell is 0,...,7
        return self.cell_list[cell]

    # play step(game)
    def step(self, action):
        """
        step function should take as input action and return:
        observation - current env.score table
        reward - cooperation level gained in this step
        terminated - whether the game is over or not
        info - None
        """
        # set game_counter
        self.games_counter += 1
        # if self.games_counter % 1000 == 0:
        #     print('self.games_counter', self.games_counter)
        # get action of the step
        change, cell = self.action_to_change_cell(action)

        # update env.score table:
        self.update_score_table(change, cell)

        # # interaction with low_env:
        dqnAgent0 = DQNAgent(self.low_env)  # self.episode_length = 1000
        dqnAgent1 = DQNAgent(self.low_env)
        dqnAgent0.train_double_dqn(self.low_env, dqnAgent1, num_episodes=self.low_level_episode_length)

        # set rewards as a function of the steps
        reward = self.get_reward()

        # save game history
        self.history.append(action)

        # set in the obs{} dict history for every agent (opponent hist)
        state = self.low_env.score.reshape(1, -1)[0]

        # episode ends when reach meta_episode_len
        done = (self.games_counter % self.meta_episode_len == 0)

        # print('done', done)
        info = {'r': reward}
        # print('self.games_counter: ', self.games_counter)
        return state, reward, done, info

    def reset(self, seed=None, options=None):
        # reset games_counter:
        # self.games_counter = 0

        # reset done:
        self.done = False

        # reset env.score table:
        self.low_env = PdeSingleAgentEnv(self.memory_size)
        obs = self.low_env.score.reshape(1, -1)[0]

        return obs

    def render(self):
        # Todo
        return None

    @property
    def action_space(self):
        # action_space can be rather 0 - 23 - Cooperate
        return Discrete(self.n_actions)

    @property
    def observation_space(self):
        # observation_space can be any score table
        return Box(low=-10, high=10, shape=(8, ), dtype=int)

    @staticmethod
    def plot_durations(episode_rewards, show_result=False):
        plt.figure(1)
        reward_steps = torch.tensor(episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(reward_steps.numpy())
        # Take 100 episode averages and plot them too
        if len(reward_steps) >= 1:
            means = reward_steps.unfold(0, 1, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(0), means))
            plt.plot(means.numpy())

        # plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
print (device_lib.list_local_devices())

if __name__ == '__main__':

    Meta_env = PdeVecEnv(memory_size=2)
    # meta_agent = MetaDQNAgent(env)
    num_envs = 10
    env = make_vec_env(PdeVecEnv, n_envs=num_envs, env_kwargs={"memory_size": 2})  # vec_env_cls=SubprocVecEnv
    # Create the environment
    # env = Meta_env
    env.reset()

    model = PPO('MlpPolicy', env, verbose=1, n_steps=64, tensorboard_log="./tmp/ppo_models/")
    # model.load:
    print('model created')
    # buffer_length = len(model.get_buffer())
    model.learn(total_timesteps=20000, progress_bar=True, tb_log_name="first_run")  # total_time_steps=depends on n_steps
    print('model learned')
    episodes = 10
    episode_rewards = []

    log_dir = "/tmp/ppo_models/"
    model.save(log_dir + "ppo_vec_env1")


    for _ in range(10):
        env = Meta_env
        obs = env.reset()
        total_rewards = 0

        for i in range(Meta_env.meta_episode_len):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            total_rewards += rewards
        print('episode rewards: ', total_rewards, 'ending table', obs)
        episode_rewards.append(total_rewards)
        # env.render()

    env.plot_durations(episode_rewards, show_result=True)

    env.close()

    # for ep in range(episodes):
    #
    #     # env should be changed back if used vectorized training
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         action, _states = model.predict(obs)
    #         obs, rewards, done, info = env.step(action)
    #         # env.render()
    #         print(rewards)
    #         print('episode: ', ep)
    #         episode_rewards.append(rewards)
    #
    # env.plot_durations(episode_rewards, show_result=True)
    #
    # env.close()



# two error to ask assaf:
# 1. when I run the code with subprocvecenv I get an error: ....


# next steps:
# 1. run the code with vectorized env - mainly check whether the code works
# 2. map low level reward function to a dictionary of 'states' and provide rewards
