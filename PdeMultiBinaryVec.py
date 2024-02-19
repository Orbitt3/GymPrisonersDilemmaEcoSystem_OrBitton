# class of learner which can change agents rewards and wants to maximize cooperation
import numpy as np
import random
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
import matplotlib
from PdeSingleAgentEnv import PdeSingleAgentEnv
import torch
from dqnAgent import DQNAgent
from dqnAgent import plot_cooperation_levels, plot_cooperation_levels2
import os

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C
from gym.spaces import MultiBinary
from CustomCallback import CustomCallback

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class PdeMultiBinaryVec (gym.Env):
    # init env, set memory size, game rewards, history length
    def __init__(self, low_level_episode_length, memory_size=2, symmetrical=True):
        self.env_name = 'Meta_PDE_MultiBinary'

        # low level params:
        self.memory_size = memory_size
        self.low_env = PdeSingleAgentEnv(memory_size)
        self.low_level_episode_length = low_level_episode_length
        self.dqnAgent0 = DQNAgent(self.low_env)  # self.episode_length = 1000
        self.dqnAgent1 = DQNAgent(self.low_env)

        # high level params:
        self.change_options = 2  # 3 for -1,0,1
        self.n_actions = int(self.low_env.score.reshape(8, -1).shape[0])*self.change_options  # 8X1[0]
        # self.n_observations = self.low_env.score.reshape(8, -1).shape[0]  # 8X1[0]
        self.n_observations = 13 #np.concatenate((self.low_env.score.reshape(8, -1),
                                              # self.compute_equations().reshape(5, -1)), axis=0).shape[0]

        self.change_factor = 1
        self.symmetrical = symmetrical

        self.history = []
        self.meta_episode_len = 4
        self.cell_list = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
        self.bounds = [-0, 9]
        self.reward_memory = np.zeros(int(10e6))

        self.games_counter = 0
        self.done = False

    def get_reward(self):
        # count rewards for meta agent:
        meta_hist = self.low_env.last10_history  # deque of 100 length 2 lists
        meta_hist = [item for sublist in meta_hist for item in sublist if sublist[0] == 1 and sublist[1] == 1]  # flatten list
        meta_reward = meta_hist.count(1)  # count number of co-op actions in game hist
        if meta_reward == 0:
            meta_reward = -50
        # meta_reward = self.low_env.score[3, 1] + self.low_env.score[3, 0]  # get meta reward from score table
        return meta_reward

    def compute_equations(self):
        table = self.low_env.score.copy()
        P = table[0, 0]
        T = table[1, 1]
        S = table[1, 0]
        R = table[3, 0]
        vec = np.zeros(5)
        vec[0] = (R-P-1)*2
        vec[1] = R-S-1
        vec[2] = 2*R-T-S-1
        vec[3] = (T-R-1)*2
        vec[4] = (P-S-1)*0.5
        return vec

    def check_valid_bounds(self, change, cell):
        new_val = self.low_env.score[cell[0], cell[1]] + change
        if new_val <= self.bounds[1] and new_val >= self.bounds[0]:
            return True
        return False

    @staticmethod
    def return_opposite(bit):
        if bit == 0:
            return 1
        return 0

    @staticmethod
    def get_twin_cell(cell):
        if cell[0] == 1:
            return cell[0]+1, PdeMultiBinaryVec.return_opposite(cell[1])
        elif cell[0] == 2:
            return cell[0]-1, PdeMultiBinaryVec.return_opposite(cell[1])
        if cell[0] == 0 or cell[0] == 3:
            return cell[0], PdeMultiBinaryVec.return_opposite(cell[1])

    def update_score_table(self, change, cell, symmetrical=True):
        if self.check_valid_bounds(change, cell):
            self.low_env.score[cell[0], cell[1]] += change
            if symmetrical:
                twin_cell = self.get_twin_cell(cell)
                self.low_env.score[twin_cell[0], twin_cell[1]] += change

    def action_to_change_cell(self, selection):
        """
        get selection from 0 to 23(num_actions) and returns action and cell
        """
        change = selection % self.change_options  # 0,1 #,2
        cell = int(selection / self.change_options)  # 0,...,7

        change = self.map_argmax_change_to_change(change)
        cell = self.map_argmax_cell_to_cell(cell)
        return change, cell

    def map_argmax_change_to_change(self, change):
        # modified_change = change - 1
        if change == 0:
            modified_change = -1
        else:
            modified_change = 1
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
        for idx, act in enumerate(action):
            if act == 1:  # if the action at the current index is selected
                change, cell = self.action_to_change_cell(idx)
                # print('idx', idx, 'act', act, 'change', change, 'cell', cell)
                # print('change', change, 'cell', cell)

                self.update_score_table(change, cell, symmetrical=self.symmetrical)

        # update env.score table:
        # self.update_score_table(change, cell)

        if self.games_counter % 10000==0:
        # # interaction with low_env:
            print('len of history', len(self.history))
            print(self.low_env.score)
            dqnAgent0 = DQNAgent(self.low_env)  # self.episode_length = 100
            dqnAgent1 = DQNAgent(self.low_env)
        # dqnAgent0.train_double_dqn(self.low_env, dqnAgent1, num_episodes=self.low_level_episode_length)

        self.dqnAgent0.train_double_dqn(self.low_env, self.dqnAgent1, num_episodes=self.low_level_episode_length)

        # set rewards as a function of the steps
        equations_values = self.compute_equations()
        equations_reward = equations_values.sum()  # normalize  to [-1,1]
        reward = self.get_reward()*4 + equations_reward
        # reward = self.get_reward()

        # save game history
        self.history.append(action)

        # set in the obs{} dict history for every agent (opponent hist)
        state = self.low_env.score.reshape(1, -1)[0]

        # add equations to state
        vec = self.compute_equations()
        state = np.concatenate((state, vec), axis=0)

        # episode ends when reach meta_episode_len
        done = (self.games_counter % self.meta_episode_len == 0)
        # done = False
        # print('done', done)
        info = {'r': reward}
        print('self.games_counter: ', self.games_counter)
        return state, reward, done, info

    def reset(self, seed=None, options=None):
        # reset games_counter:
        # self.games_counter = 0

        # reset done:
        self.done = False

        # reset env.score table:
        self.low_env = PdeSingleAgentEnv(self.memory_size)
        obs = self.low_env.score.reshape(1, -1)[0]
        obs = np.concatenate((obs, self.compute_equations()), axis=0)
        return obs

    def render(self):
        # Todo
        return None

    # @property
    # def action_space(self):
    #     # action_space can be rather 0 - 23 - Cooperate
    #     return Discrete(self.n_actions)

    @property
    def action_space(self):
        return MultiBinary(self.n_actions)

    @property
    def observation_space(self):
        # observation_space can be any score table
        return Box(low=-10, high=10, shape=(13, ), dtype=int)



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

# if __name__ == '__main__':
#
#     Meta_env = PdeMultiBinaryVec(memory_size=2)
#     # meta_agent = MetaDQNAgent(env)
#     num_envs = 4
#     env = make_vec_env(PdeMultiBinaryVec, n_envs=num_envs, env_kwargs={"memory_size": 2})  # vec_env_cls=SubprocVecEnv
#     # Create the environment
#     # env = Meta_env
#     env.reset()
#
#     model = PPO('MlpPolicy',  env, verbose=1, n_steps=128, tensorboard_log="./tmp/ppo_models/")
#     # model.load:
#     print('model created')
#     # buffer_length = len(model.get_buffer())
#     callback = CustomCallback(env=env, verbose=1)
#     model.learn(total_timesteps=5000, progress_bar=True, tb_log_name="first_run", callback=callback)  # total_time_steps=depends on n_steps
#     print('model learned')
#     episodes = 4
#     episode_rewards = []
#
#     log_dir = "/tmp/ppo_models/"
#     model.save(log_dir + "ppo_vec_env1")
#
#     histories = []
#     for _ in range(2):
#         env = Meta_env
#         obs = env.reset()
#         total_rewards = 0
#
#         for i in range(Meta_env.meta_episode_len):
#             action, _states = model.predict(obs)
#             print('action: ', action, 'obs: ', obs)
#             obs, rewards, dones, info = env.step(action)
#         total_rewards += rewards
#         print('episode rewards: ', total_rewards, 'ending table', obs)
#         episode_rewards.append(total_rewards)
#         # env.render()
#
#         env.plot_durations(episode_rewards, show_result=True)
#         ############################
#         dqnAgent = Meta_env.dqnAgent0
#         dqn_op_agent = Meta_env.dqnAgent1
#
#         # plot block:
#         dqnAgent.train_double_dqn(Meta_env.low_env, dqn_op_agent, num_episodes=1500)
#         # plot cooperation level
#         history = Meta_env.low_env.total_history
#         histories.append(history)
#         avg_cooperation = plot_cooperation_levels(histories, Meta_env.low_env.score.reshape(8, -1))
#     plt.show()
#     # dqnAgent.test_double_dqn(Meta_env.low_env, dqn_op_agent, num_games=1000)
#     env.close()

if __name__ == '__main__':
    histories = []
    ending_tables = []
    # New loop for the different low_level_episode_length values
    for episode_length in [10, 50, 100]:

        Meta_env = PdeMultiBinaryVec(memory_size=2, low_level_episode_length=episode_length)
        # meta_agent = MetaDQNAgent(env)
        num_envs = 4
        env = make_vec_env(PdeMultiBinaryVec, n_envs=num_envs,
                           env_kwargs={"memory_size": 2, "low_level_episode_length": episode_length})  # vec_env_cls=SubprocVecEnv
        # Create the environment
        # env = Meta_env
        env.reset()

        model = PPO('MlpPolicy', env, verbose=1, n_steps=128, tensorboard_log="./tmp/ppo_models/")
        # model.load:
        print('model created')
        # buffer_length = len(model.get_buffer())
        callback = CustomCallback(env=env, verbose=1)
        model.learn(total_timesteps=60000, progress_bar=True, tb_log_name="first_run",
                    callback=callback)  # total_time_steps=depends on n_steps
        print('model learned')
        episodes = 4
        episode_rewards = []

        log_dir = "/tmp/ppo_models/"
        model.save(log_dir + "ppo_vec_env1")


        for _ in range(1):
            env = Meta_env
            obs = env.reset()
            total_rewards = 0

            for i in range(Meta_env.meta_episode_len):
                action, _states = model.predict(obs)
                print('action: ', action, 'obs: ', obs)
                obs, rewards, dones, info = env.step(action)
            total_rewards += rewards
            print('episode rewards: ', total_rewards, 'ending table', obs)
            episode_rewards.append(total_rewards)
            ending_tables.append(Meta_env.low_env.score.reshape(8, -1))
            # env.render()
            # env.plot_durations(episode_rewards, show_result=True)
            ############################

            dqnAgent = Meta_env.dqnAgent0
            dqn_op_agent = Meta_env.dqnAgent1

            # plot block:
            dqnAgent.train_double_dqn(Meta_env.low_env, dqn_op_agent, num_episodes=1500)
            # plot cooperation level
            history = Meta_env.low_env.total_history
            histories.append(history)

    # Plot cooperation levels outside the loops
    avg_cooperation = plot_cooperation_levels(histories, Meta_env.low_env.score.reshape(8, -1))
    avg_cooperation = plot_cooperation_levels2(histories, Meta_env.low_env.score.reshape(8, -1))

    plt.show()
    print(ending_tables)
    # dqnAgent.test_double_dqn(Meta_env.low_env, dqn_op_agent, num_games=1000)
    env.close()




