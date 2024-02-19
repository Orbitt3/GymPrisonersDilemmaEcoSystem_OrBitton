
# class of learner which can change agents rewards and wants to maximize cooperation
import numpy as np
import random
from PdeMetaAgentEnv import PdeMetaAgentEnv
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
from agent import Agent
import tensorflow as tf
from PdeSingleAgentEnv import PdeSingleAgentEnv


class MetaAgent:
    def __init__(self, env, alpha=0.3, gamma=0.99, epsilon=1, epochs=5000, agent_memory=4):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.total_rewards = 0
        self.immediate_reward = 0
        self.agent_memory = agent_memory
        self.meta_q_table = env.meta_q_table()
        self.epsilon_decay = (self.epsilon/(epochs*0.35))*1000


    @staticmethod
    def state_to_row(state):
        """
        :param state: np.array of the form - env.history
        :return: row number as a function of this array
        """
        agent0_hist, agent1_hist = state[0], state[1]
        total_state = list(agent0_hist) + list(agent1_hist)
        total_state = list(reversed(total_state))

        row_number = 0
        for i, value in enumerate(total_state):
            row_number += (2 ** i) * int(total_state[i])
        return row_number

    @staticmethod
    def action_to_column(action):
        """
        :param action: tuple of the form - (change, cell)
        :return: column number as a function of this tuple
        """
        change = action[0]  # -1,0,1
        cell = action[1]  # cell from the shape (x,y) from 4x2 matrix
        change_index = change*8 + 8  # 0,8,16
        cell_index = 2*cell[0] + cell[1]  # values of 0-7
        return change_index + cell_index

    @staticmethod
    def column_to_action(column):
        """
        :param column (row(0-15) x column(0-23))
        :return: action: (change, cell)
        """
        change = int(column/8) - 1  # change: (-1,0,1)
        cell = [0, 0]
        temp = column - (8*(change+1))  # temp is 0-7 mapping
        cell[0] = int(temp/2)
        cell[1] = column % 2  # cell[1] can be 0/1
        action = (change, (cell[0], cell[1]))  # tuple of (change, (cell[x],cell[y])
        return action

    @staticmethod
    def sample_action_space():
        change = random.randint(-1, 1)
        row = random.randint(0, 3)
        column = random.randint(0, 1)
        cell = (row, column)
        cell = (3, 0)  # should be erased after Meta_agent will work properly
        action = (change, cell)
        return action

    # get observation and return action
    def predict(self, state, deterministic=False):
        row = MetaAgent.state_to_row(state)
        cell_list = [self.meta_q_table[row][6], self.meta_q_table[row][14], self.meta_q_table[row][22]]
        bool_state = True
        if 0 in cell_list:
        # if self.epochs <= 10000:
            action = MetaAgent.sample_action_space()
            print(f' 0 in cell actions: {action}')
            print(f' state, cell list:{state} {cell_list}')

        else:
            action_value_list = self.meta_q_table[row]
            # print(action_value_list)
            # action_value_list += self.meta_q_table[row][14:15]
            # action_value_list += self.meta_q_table[row][22:23]
            column = np.nanargmax(action_value_list)
            action = MetaAgent.column_to_action(column)
            print(f' Rational move: column {column}, action {action}')

            if (~deterministic) & (random.uniform(0, 1) < self.epsilon):
                action = MetaAgent.sample_action_space()
                print(f' Stochastic move: column {column}, action {action}')

        return action

    # store
    def store(self, state, action, reward, next_state):
        column = self.action_to_column(action)
        if column == 0:
            print('state, action, reward, next_state')
            print(state, action, reward, next_state)
        # sum total rewards:
        self.total_rewards += reward
        self.immediate_reward = reward

        # get q_table old value  #####
        row = MetaAgent.state_to_row(state)
        old_value = self.meta_q_table[row, column]

        # update q_table
        next_row = MetaAgent.state_to_row(next_state)
        next_max = np.nanmax(self.meta_q_table[next_row])  # check nan-max
        new_value = (1-self.alpha) * old_value + self.alpha*(reward + self.gamma*next_max)
        self.meta_q_table[row, column] = new_value

        # return new_value-old_value  # returns Delta_Q_table

    @staticmethod
    def learn_meta_agent(env, num_steps, agent0, agent1, meta_agent):
        """
        agents will learn every step, meta_agent will learn every episode? 5000?
         (agents epsilon will reset)
        agents inputs - opponents and their own history
        meta_agent input - score board
        """
        meta_epoch = 1000
        obs = env.reset()  # reset  Pde env
        state0 = (obs["agent-0"], obs["agent-1"])
        state1 = (obs["agent-1"], obs["agent-0"])
        next_meta_state = env.history
        done = False
        cooperation_count = [0, 0]
        predict_count = 0
        store_count = 0
        # in our case num_steps = num epochs
        for step in tqdm(range(num_steps)):

            deterministic = False  # Reset deterministic value to false
            meta_agent.epochs = step
            # if there is not enough actions_history, make a random action
            if step <= 2**agent0.agent_memory:
                action0 = env.action_space.sample()
                action1 = env.action_space.sample()
                meta_action = (0, (0, 0))

            else:
                if step % meta_epoch <= 4:
                    deterministic = True  # make sure env.history steps are not random (10>2)
                # set next states and get actions:
                state0 = next_state0
                state1 = next_state1
                meta_state = next_meta_state
                # print(meta_state)

                # get actions:
                action0 = agent0.predict(env, state0, deterministic)
                action1 = agent1.predict(env, state1, deterministic)

                if step % meta_epoch == 0:  # make change every 1000 steps
                    meta_action = meta_agent.predict(meta_state)  # meta_state = env.history

                # count cooperation times:
                if action0 == action1:
                    if action0 == 1:
                        cooperation_count[0] += 1
                    else:
                        cooperation_count[1] += 1
            actions = {"agent-0": action0, "agent-1": action1, "meta_agent": meta_action}
            # print(f' actions: {actions}')

            # apply the action and interact with the environment
            next_state0, reward, done, _ = env.step(actions)  # next = op_hist
            next_state1 = (next_state0[1], next_state0[0])  # reversed tuple
            reward["meta_agent"] = (cooperation_count[0] / (step + 1)) - meta_agent.immediate_reward
            if reward['meta_agent'] < 0:
                reward['meta_agent'] = np.finfo(float).eps

            # update meta_agent values only if it's 'epoch time'
            if (step % meta_epoch == 0) and (step != 0):
                next_meta_state = env.history
                meta_agent.store(meta_state, meta_action, reward["meta_agent"], next_meta_state)
                agent0.epsilon = 0.1  # reset epsilon in-order to get new exploration period
                agent1.epsilon = 0.1
                if meta_agent.epsilon > 0.1:
                    meta_agent.epsilon -= meta_agent.epsilon_decay

            else:
                meta_action = (0, (0, 0))  # reset actions for meta agent

            # fill-up personal Q-table
            agent0.store(state0, action0, reward["agent-0"], next_state0)
            agent1.store(state1, action1, reward["agent-1"], next_state1)

            # Render the env
            env.write_summary(episode=step, agent0=agent0, agent1=agent1, meta_agent=meta_agent)
            env.render()

            # if done:
            #     env.reset()

            if agent0.epsilon > 0.01:
                agent0.epsilon -= agent0.epsilon_decay
                agent1.epsilon -= agent1.epsilon_decay

        return agent0.q_table, agent1.q_table, meta_agent.meta_q_table, cooperation_count

    def compute_missing_actions(self):
        """ returns number of cells with value == 0"""
        arr_cell_size = self.meta_q_table.shape[0]*self.meta_q_table.shape[1]
        zeros_count = arr_cell_size - np.count_nonzero(self.meta_q_table == 0.0)
        return zeros_count
