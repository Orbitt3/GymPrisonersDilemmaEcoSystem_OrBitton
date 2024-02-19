import numpy as np
import random
from gym_example import Pde
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
import math
import tensorflow as tf


class Agent:
    # initial an agent with certain amount of memory
    def __init__(self, env, alpha=0.2, gamma=0.9, epsilon=1, epochs=5000, agent_memory=2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.total_rewards = 0
        self.immediate_reward = 0
        self.agent_memory = agent_memory
        # self.epsilon_decay = self.epsilon/(epochs*0.35)
        self.epsilon_decay = 2/(epochs*0.05)

        self.delta_q = 1

        # set q_table size where q(state,action) and state = (my_state, opponent_state)
        rows = (2**self.agent_memory)**2
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

    def softmax_with_temp(self, action_value_list, temperature):
        # print('action_value_list', action_value_list, 'temperature', temperature)
        # temperature_values_list = [x / temperature for x in action_value_list]
        x = action_value_list / temperature

        # bottom = sum([math.exp(x) for x in temperature_values_list])
        e_x = np.exp(x - np.max(x))
        # softmax_with_temp = [math.exp(x) / bottom for x in temperature_values_list]
        softmax_with_temp = e_x / e_x.sum()
        return softmax_with_temp


    # get observation and return action
    def predict(self, env, state, deterministic=False):
        row = Agent.state_to_row(state)
        if 0 in self.q_table[row]:
            action = env.action_space.sample()
        else:
            action_value_list = self.q_table[row]
            probability_list = self.softmax_with_temp(action_value_list, self.epsilon)
            action = np.random.choice([0,1], p=probability_list)
            # action = np.nanargmax(action_value_list)
            # if (~deterministic) & (random.uniform(0, 1) < self.epsilon):
            #     action = env.action_space.sample()
        return action

    # store
    def store(self, state, action, reward, next_state):
        # print('state', state[0], type(state[0]))
        # sum total rewards:
        self.total_rewards += reward
        self.immediate_reward = reward

        # get q_table old value
        row = Agent.state_to_row(state)
        old_value = self.q_table[row, action]

        # update q_table
        next_row = Agent.state_to_row(next_state)
        next_max = np.nanmax(self.q_table[next_row])  # check nanmax
        new_value = (1-self.alpha) * old_value + self.alpha*(reward + self.gamma*next_max)
        self.q_table[row, action] = new_value

        self.delta_q = new_value-old_value  # returns Delta_Q_table

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
            if step <= 2**agent0.agent_memory:
                action0 = env.action_space.sample()
                action1 = env.action_space.sample()

            else:
                # set next states:
                state0 = next_state0
                state1 = next_state1

                # get actions:
                action0 = agent0.predict(env, state0, deterministic)
                action1 = agent1.predict(env, state1, deterministic)

                # count cooperation times:
                # if action0 == action1:
                if True:
                    if action0 == 1 or action1 == 1:
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
            # env.write_summary(episode=step, agent0=agent0, agent1=agent1)
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

    @staticmethod
    def plotter(agent0, reward_0, agent1, reward_1, step):
        print('Game:', step, 'Score_agent0:', agent0.total_rewards,
              'Score_agent1:', agent1.total_rewards)
        plot_scores = [[], []]
        plot_mean_scores = [[], []]

        plot_scores[0].append(reward_0)
        plot_scores[1].append(reward_1)

        mean_score0 = agent0.total_rewards / step
        mean_score1 = agent1.total_rewards / step

        plot_mean_scores[0].append(mean_score0)
        plot_mean_scores[1].append(mean_score1)

        Agent.plot(plot_scores[0], plot_mean_scores[0])
        # Agent.plot(plot_scores[1], plot_mean_scores[1])

    @staticmethod
    def plot(scores, mean_scores):
        plt.ion()
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        plt.show(block=False)
        plt.pause(.1)

    def compute_missing_actions(self):
        """ returns number of cells with value == 0"""
        arr_cell_size = self.q_table.shape[0]*self.q_table.shape[1]
        zeros_count = arr_cell_size - np.count_nonzero(self.q_table == 0.0)
        return zeros_count
