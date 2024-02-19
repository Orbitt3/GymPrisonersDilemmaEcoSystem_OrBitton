import numpy as np
import gym
from gym.spaces import Discrete, Box
from collections import deque
import tensorflow as tf
import datetime
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# fileName = 'runs'
fileName = "runs/run/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class PdeSingleAgentEnv(gym.Env):
    # init env, set memory size, game rewards, history length
    def __init__(self, memory_size):
        self.env_name = 'PDE'
        self.memory_size = memory_size
        self.score = np.array([[1, 1], [0, 2], [2, 0], [0, 0]])
        self.history = deque(maxlen=memory_size)
        self.last100_history = deque(maxlen=100)
        self.last10_history = deque(maxlen=10)
        self.total_history = []

        self.games_counter = 0
        self.done = False
        self.summary_writer = tf.summary.create_file_writer(fileName)

    # play step(game)
    def step(self, actions):
        # set game_counter
        self.games_counter += 1

        # get actions of the step
        actions_list = list(actions.values())
        row = PdeSingleAgentEnv.actions_list_to_row(actions_list)

        # set rewards as a function of the steps
        rewards = {"agent-0": self.score[row][0], "agent-1": self.score[row][1]}

        # save game history
        self.history.append(actions_list)
        self.last10_history.append(actions_list)
        self.last100_history.append(actions_list)
        self.total_history.append(actions_list)

        # set in the obs{} dict history for every agent (opponent hist)
        state = ([s[0] for s in self.history], [s[1] for s in self.history])

        # step always ends in one step in PD
        done = {"agent-0": False, "agent-1": False}

        return state, rewards, done, None

    def reset(self, seed=None, options=None):
        # set deque in the size of memory_size
        self.history = deque(maxlen=self.memory_size)
        self.done = False
        # fill the deque with 'memory_size' random observations
        obs = {"agent-0": self.observation_space.sample(),
               "agent-1": self.observation_space.sample()}
        # print('obs[agent-0]: ', obs['agent-0'], 'obs[agent-0].shape: ', obs['agent-0'].shape, type(obs['agent-0']))

        actions_list = np.concatenate((obs['agent-0'].reshape(-1), obs['agent-1'].reshape(-1)), axis=0)
        # print('actions_list: ', actions_list, 'actions_list.shape: ', actions_list.shape)
        self.history.append(actions_list)
        return obs

    # should render to screen the games
    def render(self):
        # Todo
        return None

    @property
    def action_space(self):
        # action_space can be rather 0 - Defect or 1 - Cooperate
        return Discrete(2)

    @property
    def observation_space(self):
        # observation_space can be 0/1, shape - memory_sizeX1, type - int
        return Box(low=0, high=1, shape=(self.memory_size, 1), dtype=int)

    @staticmethod
    def actions_list_to_row(actions_list):
        row_number = 0
        for i, value in enumerate(actions_list):
            row_number += (2 ** i) * int(actions_list[i])
        return row_number

    @staticmethod
    def compute_cooperation_level(history, last_archi_epoch=0):
        """ should return mutual cooperation lvl of last x steps"""
        count = 0
        max_plays = 2 * len(history)
        for ls in history:
            count += ls.count(1)
        return count/max_plays

    @staticmethod
    def compute_missing_actions(agent0, agent1):
        agent0_count = agent0.compute_missing_actions()
        agent1_count = agent1.compute_missing_actions()
        return agent0_count + agent1_count

    def write_summary(self, episode, agent0, agent1):
        return None
        # compute values
        rewards = agent0.total_rewards + agent1.total_rewards
        epsilon = agent0.epsilon
        co_op_level = PdeSingleAgentEnv.compute_cooperation_level(self.history, episode)

        # compute total missing actions:
        q_table_size = agent0.q_table.shape[0] * agent0.q_table.shape[1]
        total_missing_actions = self.compute_missing_actions(agent0, agent1)
        q_table_miss_actions = total_missing_actions / (2*q_table_size)
        mean_delta_q1_q0 = np.mean(np.absolute(agent0.q_table-agent1.q_table))

        with self.summary_writer.as_default():
            # add scalar graphs
            tf.summary.scalar(name="Cooperation level", data=co_op_level, step=episode)
            # tf.summary.scalar(name="Delta_Q_agent0", data=agent0.delta_q, step=episode)
            # tf.summary.scalar(name="Delta_Q_agent1", data=agent1.delta_q, step=episode)
            # tf.summary.scalar(name="Mean_Delta_Q_Between_agents", data=mean_delta_q1_q0, step=episode)

            # tf.summary.scalar(name="agent-0_reward", data=agent0.immediate_reward, step=episode)
            # tf.summary.scalar(name="agent-1_reward", data=agent1.immediate_reward, step=episode)

            # tf.summary.scalar(name="Q-Table Missing Values", data=q_table_miss_actions, step=episode)
            # tf.summary.scalar(name="Epsilon", data=epsilon, step=episode)
            # tf.summary.scalar(name="Total reward", data=rewards, step=episode)

        if (episode % 100) == 0:
            self.summary_writer.flush()

