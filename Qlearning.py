import numpy as np
import random
from gym_example import Pde


class QLearning:
    # does alpha and gamma have values if the game has only one step?
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1, epochs=5000, agent_memory=2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.agent_memory = agent_memory

        # set agents own q-table with size agent_memory*action_space
        agent_history = 2**self.agent_memory
        opponent_history = 2**self.agent_memory
        self.q_table = np.zeros((agent_history, opponent_history, env.action_space.n))

    # set agents own q-table with size agent_memory*action_space
    # @property
    # def q_Table(self):
    #     q_table = np.zeros((self.agent_memory, Env.action_space.n)).reshape(self.agent_memory, Env.action_space.n)
    #     # q_table shape - memory-size*2
    #     return q_table

    @staticmethod
    def state_to_row(state):
        row_number = 0
        for i, value in enumerate(state):
            row_number += (2 ** i) * int(state[i])
        return row_number

    def predict(self, env, my_obs, my_hist, q_table, deterministic=False):
        matrix = QLearning.state_to_row(my_hist)
        row = QLearning.state_to_row(my_obs)
        if 0 in q_table[matrix, row]:
            action = env.action_space.sample()
        else:
            action_value_list = self.q_table[matrix, row]
            action = np.nanargmax(action_value_list)
            if (~deterministic) & (random.uniform(0, 1) < self.epsilon):
                action = env.action_space.sample()
        return action

    def store(self, my_obs, my_hist, action, reward, next_obs, next_hist):  # my_obs = opponent history, op_obs = my history

        matrix = QLearning.state_to_row(state=my_hist)
        row = QLearning.state_to_row(state=my_obs)
        old_value = self.q_table[matrix, row, action]
        # update q_table
        next_matrix = QLearning.state_to_row(state=next_hist)
        next_row = QLearning.state_to_row(state=next_obs)

        next_max = np.nanmax(self.q_table[next_matrix, next_row])
        new_value = (1-self.alpha)*old_value + self.alpha * (reward + self.gamma*next_max)
        self.q_table[matrix, row, action] = new_value

    @staticmethod
    def learn(env, num_steps, agent0, agent1, memory_size):
        # reset env
        obs = env.reset()
        epochs, penalties, reward, = 0, 0, 0
        done = False
        cooperation_count = [0, 0]

        # in our case num_steps = num epochs
        for step in range(num_steps):
            # if action there is not enough actions_history, make a random action
            if step <= 100:
                action0 = env.action_space.sample()
                action1 = env.action_space.sample()
                # print('stochastic move')

            else:
                action0 = agent0.predict(env, obs["agent-0"], obs["agent-1"], agent0.q_table)
                action1 = agent1.predict(env, obs["agent-1"], obs["agent-0"], agent1.q_table)
                if action0 == action1:
                    if action0 == 1:
                        cooperation_count[0] += 1
                    else:
                        cooperation_count[1] += 1
            actions = {"agent-0": action0, "agent-1": action1}
            # apply the action and interact with the environment
            next_state, reward, done, _ = env.step(actions)
            # fill-up personal Q-table
            agent0.store(obs["agent-0"], obs["agent-1"], action0, reward["agent-0"], next_state["agent-0"], next_state["agent-1"])
            agent1.store(obs["agent-1"], obs["agent-0"], action1, reward["agent-1"], next_state["agent-1"], next_state["agent-0"])

            obs = next_state
            # Render the env
            env.render()

            # # If the episode is up, then start another one
            # if done:
            #     env.reset()
            if agent0.epsilon > 0.05:
                agent0.epsilon *= 0.9999
                agent1.epsilon *= 0.9999
            # if step % 1000 == 0:
            #     print(f'epoch num: {step} epsilon: {agent0.epsilon}')
        return agent0.q_table, agent1.q_table, cooperation_count
