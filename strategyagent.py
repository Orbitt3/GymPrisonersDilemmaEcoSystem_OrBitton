import numpy as np
import random
from gym_example import Pde
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
import tensorflow as tf
from agent import Agent


class StrategyAgent(Agent):
    # initial an agent with certain amount of memory
    def __init__(self, env, alpha=0.3, gamma=0.9, epsilon=1, epochs=5000, agent_memory=2):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.total_rewards = 0
        self.immediate_reward = 0
        self.agent_memory = agent_memory
        self.epsilon_decay = self.epsilon / (epochs * 0.35)
        self.delta_q = 1
        self.strategy = None

        # set q_table size where q(state,action) and state = (my_state, opponent_state)
        rows = (2 ** self.agent_memory) ** 2
        self.q_table = np.zeros((rows, env.action_space.n))

    @staticmethod
    def state_to_row(state):
        """
        :param state: tuple of the form (my_hist, op_hist)
        :return: row number as a function of this tuple
        """
        my_hist, op_hist = state[0], state[1]
        total_state = list(my_hist) + list(op_hist)
        total_state = list(reversed(total_state))

        row_number = 0
        for i, value in enumerate(total_state):
            row_number += (2 ** i) * int(total_state[i])
        return row_number

    # get observation and return action
    def predict(self, env, state, deterministic=False):
        action = self.strategy(state)
        return action

    # store
    def store(self, state, action, reward, next_state):
        # sum total rewards:
        self.total_rewards += reward
        self.immediate_reward = reward

        # get q_table old value
        row = Agent.state_to_row(state)
        old_value = self.q_table[row, action]

        # update q_table
        next_row = Agent.state_to_row(next_state)
        next_max = np.nanmax(self.q_table[next_row])  # check nanmax
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[row, action] = new_value
        self.delta_q = new_value - old_value  # returns Delta_Q_table

    @staticmethod
    def learn_as_single_agent(env, num_steps, agent0, agent1):
        obs = env.reset()  # reset env
        state0 = (obs["agent-0"], obs["agent-1"])
        state1 = (obs["agent-1"], obs["agent-0"])
        # epochs, penalties, reward, = 0, 0, 0
        done = False
        deterministic = False
        cooperation_count = [0, 0]

        # in our case num_steps = num epochs
        for step in tqdm(range(num_steps)):
            # if there is not enough actions_history, make a random action
            if step <= 4:
                action0 = env.action_space.sample()
                action1 = env.action_space.sample()

            else:
                state0 = next_state0
                state1 = next_state1
                action0 = agent0.predict(env, state0, deterministic)
                action1 = agent1.predict(env, state1, deterministic)
                if action0 == action1:
                    if action0 == 1:
                        cooperation_count[0] += 1
                    else:
                        cooperation_count[1] += 1
            actions = {"agent-0": action0, "agent-1": action1}

            # apply the action and interact with the environment
            next_state0, reward, done, _ = env.step(actions)  # next = op_hist
            next_state1 = (next_state0[1], next_state0[0])  # reversed tuple
            # fill-up personal Q-table
            agent0.store(state0, action0, reward["agent-0"], next_state0)
            agent1.store(state1, action1, reward["agent-1"], next_state1)

            # Render the env
            env.write_summary(episode=step, agent0=agent0, agent1=agent1)
            env.render()

            # if done:
            #     env.reset()

            if agent0.epsilon > 0.01:
                agent0.epsilon -= agent0.epsilon_decay
                agent1.epsilon -= agent1.epsilon_decay

        # if np.absolute(agent0.delta_q) < 0.001 and np.absolute(agent1.delta_q) < 0.001:
        #     break

        # if step > 0.8*num_steps: # play last 20% of games in a deterministic state
        #    deterministic = True

        # images = [agent0.q_table, agent1.q_table]
        # tf.summary.image('Agent_0 Q_table', images, max_outputs=2, step=0)
        # # tf.summary.image('Agent_0 Q_table', agent1.q_table)
        return agent0.q_table, agent1.q_table, cooperation_count

    def set_strategy(self, strategy):
        '''
        get strategy str and changes strategy to a method
        '''
        foo = getattr(self, strategy)
        self.strategy = foo

    def compute_missing_actions(self):
        """ returns number of cells with value == 0"""
        return 0

    def always_coop(self, state):
        return 1

    def always_defect(self, state):
        return 0

    def random_strategy(self, state):
        return random.randint(0, 1)

    # return last opponent action
    def tit_for_tat(self, state):
        state = np.array(state)
        op_hist = state[1]
        last_op_action = op_hist[-1]
        return int(last_op_action)

    # Cooperates unless defected against twice in a row
    def tit_for_two_tats(self, state):
        op_hist = state[1]
        if op_hist[-1] == 0 and op_hist[-2] == 0:
            return 0
        else:
            return 1