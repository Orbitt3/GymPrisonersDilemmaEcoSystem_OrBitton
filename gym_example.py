import numpy as np
import gym
from gym.spaces import Discrete, Box
from collections import deque


class Pde(gym.Env):

    # init env, set memory size, game rewards, history length
    def __init__(self, memory_size):
        self.env_name = 'PDE'
        self.memory_size = memory_size
        self.score = np.array([[1, 1], [0, 5], [5, 0], [3, 3]])
        self.history = deque(maxlen=memory_size)
        self.games_counter = 0
        self.done = False

    # play step(game),
    def step(self, actions):
        # get actions of the step
        actions_list = list(actions.values())
        row = Pde.actions_list_to_row(actions_list)
        # set rewards as a function of the steps
        rewards = {"agent-0": self.score[row][0], "agent-1": self.score[row][1]}

        # save game history
        self.history.append(actions_list)

        # set in the obs{} dict history for every agent (opponent hist)
        obs = {"agent-0": [s[0] for s in self.history], "agent-1": [s[1] for s in self.history]}

        # step always ends in one step in PD
        done = {"agent-0": False, "agent-1": False}

        return obs, rewards, done, None

    def reset(self, seed=None, options=None):
        # set deque in the size of memory_size
        self.history = deque(maxlen=self.memory_size)
        self.done = False
        # fill the deque with 'memory_size' random observations
        obs = {"agent-0": self.observation_space.sample(),
               "agent-1": self.observation_space.sample()}
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


