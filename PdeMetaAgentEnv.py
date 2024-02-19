import numpy as np
import gym
from gym.spaces import Discrete, Box
from collections import deque
import tensorflow as tf
import datetime
from PdeSingleAgentEnv import PdeSingleAgentEnv


# fileName = 'runs'
# fileName = "runs/meta_run/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
fileName = "runs/DQN_meta_run/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class PdeMetaAgentEnv(gym.Env):
    # init env, set memory size, game rewards, history length
    def __init__(self, memory_size):
        self.env_name = 'Meta_PDE'
        self.memory_size = memory_size
        self.score = np.array([[1, 1], [0, 5], [5, 0], [3, 3]])
        self.history = deque(maxlen=memory_size)
        self.last1000_history = deque(maxlen=1000)
        self.games_counter = 0
        self.done = False
        self.summary_writer = tf.summary.create_file_writer(fileName)

    def meta_q_table(self):
        # change can be rather -1, 0, 1 to any reward
        changes = [-1, 0, 1]
        # cells can be - from shape 4X2
        cells = self.score
        actions = len(changes) * (cells.shape[0] * cells.shape[1])
        history = (2**self.memory_size)**2
        return np.zeros((history, actions))

    # play step(game)
    def step(self, actions):  # add meta-action, add meta_reward, change score table
        meta_action = actions.pop('meta_agent')  # [change, cell(row,column)]

        # set game_counter
        self.games_counter += 1

        # get actions of the step
        actions_list = list(actions.values())
        # print(f'actions_list:{actions_list}')
        row = PdeMetaAgentEnv.actions_list_to_row(actions_list)

        # set rewards as a function of the steps
        rewards = {"agent-0": self.score[row][0], "agent-1": self.score[row][1]}

        # save game history
        self.history.append(actions_list)
        self.last1000_history.append(actions_list)

        self.score[meta_action[1][0], meta_action[1][1]] += meta_action[0]  # score_Table cell += meta_change
        self.score[3, 1] += meta_action[0]

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
        for i in range(self.memory_size):
            action0 = self.action_space.sample()
            actions = {"agent-0": action0, "agent-1": action0}
            actions_list = list(actions.values())
            # print(f'actions_list: {actions_list}')
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

    def write_summary(self, episode, agent0, agent1, meta_agent):
        # compute values
        rewards = agent0.total_rewards + agent1.total_rewards
        agents_epsilon = agent0.epsilon
        meta_epsilon = meta_agent.epsilon
        co_op_level = PdeSingleAgentEnv.compute_cooperation_level(self.history, episode)

        # compute total missing actions:
        q_table_size = agent0.q_table.shape[0] * agent0.q_table.shape[1]
        total_missing_actions = self.compute_missing_actions(agent0, agent1)
        q_table_miss_actions = total_missing_actions / (2*q_table_size)
        mean_delta_q1_q0 = np.mean(np.absolute(agent0.q_table-agent1.q_table))

        with self.summary_writer.as_default():
            # add scalar graphs
            tf.summary.scalar(name="Cooperation level", data=co_op_level, step=episode)
            tf.summary.scalar(name="Delta_Q_agent0", data=agent0.delta_q, step=episode)
            tf.summary.scalar(name="Delta_Q_agent1", data=agent1.delta_q, step=episode)
            # tf.summary.scalar(name="Mean_Delta_Q_Between_agents", data=mean_delta_q1_q0, step=episode)
            tf.summary.scalar(name="Agent_0 cooperation reward", data=self.score[3, 0], step=episode)
            tf.summary.scalar(name="Agent_1 cooperation reward", data=self.score[3, 1], step=episode)
            tf.summary.scalar(name="Meta_agent reward", data=meta_agent.immediate_reward, step=episode)

            # tf.summary.scalar(f'Agents rewards', {
            #     'Agent_0 Reward': self.score[3, 0],
            #     'Agent_0 Reward': self.score[3, 1],
            # }, episode)
            tf.summary.scalar(name="agent-0_reward", data=agent0.immediate_reward, step=episode)
            tf.summary.scalar(name="agent-1_reward", data=agent1.immediate_reward, step=episode)
            # tf.summary.scalar(name='Meta reward', data = , step= episode)

            # tf.summary.scalar(name="Q-Table Missing Values", data=q_table_miss_actions, step=episode)
            tf.summary.scalar(name="Agents Epsilon", data=agents_epsilon, step=episode)
            tf.summary.scalar(name="Meta Epsilon", data=meta_epsilon, step=episode)

            # tf.summary.scalar(name="Total reward", data=rewards, step=episode)

        if (episode % 100) == 0:
            self.summary_writer.flush()

    def write_low_level_summary(self, episode, agent0, agent1):
        # compute values
        rewards = agent0.total_rewards + agent1.total_rewards
        epsilon = agent0.epsilon
        co_op_level = PdeSingleAgentEnv.compute_cooperation_level(self.history, episode)

        # compute total missing actions:
        q_table_size = agent0.q_table.shape[0] * agent0.q_table.shape[1]
        total_missing_actions = self.compute_missing_actions(agent0, agent1)
        q_table_miss_actions = total_missing_actions / (2*q_table_size)
        mean_delta_q1_q0 = np.mean(np.absolute(agent0.q_table-agent1.q_table))

        # newfileName = "runs/DQN_meta_run/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # self.summary_writer = tf.summary.create_file_writer(newfileName)
        with self.summary_writer.as_default():
            # add scalar graphs
            tf.summary.scalar(name="Cooperation level", data=co_op_level, step=episode)
            tf.summary.scalar(name="Delta_Q_agent0", data=agent0.delta_q, step=episode)
            # tf.summary.scalar(name="Delta_Q_agent1", data=agent1.delta_q, step=episode)
            # tf.summary.scalar(name="Mean_Delta_Q_Between_agents", data=mean_delta_q1_q0, step=episode)

            tf.summary.scalar(name="agent-0_reward", data=agent0.immediate_reward, step=episode)
            # tf.summary.scalar(name="agent-1_reward", data=agent1.immediate_reward, step=episode)

            # tf.summary.scalar(name="Q-Table Missing Values", data=q_table_miss_actions, step=episode)
            tf.summary.scalar(name="Epsilon", data=epsilon, step=episode)
            # tf.summary.scalar(name="Meta_Reward", data=meta_reward, step=episode)

            # tf.summary.scalar(name="Total reward", data=rewards, step=episode)

        if (episode % 100) == 0:
            self.summary_writer.flush()
