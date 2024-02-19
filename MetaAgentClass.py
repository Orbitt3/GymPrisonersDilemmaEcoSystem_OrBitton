# class of learner which can change agents rewards and wants to maximize cooperation
import numpy as np
import random
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


# Filter tensorflow version warnings
import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu((self.layer1(x)))
        x = F.relu((self.layer2(x)))
        return self.layer3(x)


class MetaDQNAgent():
    def __init__(self, env=PdeMetaAgentEnv(memory_size=2)):

        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the AdamW optimizer
        self.BATCH_SIZE = 64
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.01
        self.EPS_DECAY = 100
        self.TAU = 0.05
        self.LR = 1e-4

        # Low agent params:
        self.low_level_episode_length = 1500
        self.low_agent_memory = 2
        self.memory_size = 2
        self.env = env
        self.episode_length = 10

        # Meta agent params:
        self.n_actions = 3  # self.env.action_space.n
        self.change_options = 3  # -a, 0, +a
        self.n_actions = int(self.env.score.reshape(8, -1).shape[0])*self.change_options  # 8X1[0]
        self.n_observations = env.score.reshape(8, -1).shape[0]  # 8X1[0]
        self.bounds = [-10, 10]
        self.episode_rewards = []
        self.steps_done = 0
        self.meta_push_pace = 1
        self.change_factor = 2
        self.meta_episode_length = 2

        # DNN params:
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(1000)

    def map_selection_to_action(self, selection):
        """
        get selection from 0 to 23(num_actions) and returns action and cell
        """
        action = selection % self.change_options  # 0,1,2
        cell = int(selection / self.change_options)  # 0,...,7
        return action, cell

    def select_action(self, state):

        self.steps_done += 1

        # Softmax policy:
        softmax = nn.Softmax(dim=1)
        action_probs = softmax(self.policy_net(state))
        dist = Categorical(action_probs)
        action = dist.sample()  # 0-23
        change, cell = self.map_selection_to_action(action)
        return torch.tensor([[change]], device=device, dtype=torch.long), cell  # arg-action is 0-2 and arg-cell is 0-7

    def plot_durations(self, show_result=False):
        plt.figure(1)
        reward_steps = torch.tensor(self.episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(reward_steps.numpy())
        # Take 100 episode averages and plot them too
        if len(reward_steps) >= 5:
            means = reward_steps.unfold(0, 5, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(4), means))
            plt.plot(means.numpy())

        # plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                      dtype=torch.bool)
        # print('batch.next_state', batch.next_state)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # print('batch.state', batch.state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # print('action_batch.shape', action_batch.shape)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def check_valid_bounds(self, change, cell):
        new_val = self.env.score[cell[0], cell[1]] + change.item()
        if new_val <= self.bounds[1] and new_val >= self.bounds[0]:
            return True
        return False

    def map_argmax_change_to_change(self, change):
        modified_change = change - 1
        modified_change = modified_change * self.change_factor
        return modified_change

    @staticmethod
    def map_argmax_cell_to_cell(cell):  # cell is 0,...,7
        cell_list = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
        return cell_list[cell]

    def update_score_table(self, change, cell):
        self.env.score[cell[0], cell[1]] += change.item()  # score_Table cell += meta_change
        # self.env.score[3, 1] += change.item()

    def meta_step(self, agent0, agent1, action, step):

        change_selection, cell_selection = action
        change = self.map_argmax_change_to_change(change_selection)  # change is -a, 0, a, but from net it is 0, 1, 2
        cell = self.map_argmax_cell_to_cell(cell_selection)
        # let meta update score table only in 100th step ##############
        meta_push_pace = 1
        if step % self.meta_push_pace != 0:
            change *= 0

        # update score table if in bounds:
        if self.check_valid_bounds(change,cell):
            self.update_score_table(change, cell)

        # let two dqn_agents play with new environment:
        agent0.train_double_dqn(self.env, agent1, num_episodes=self.low_level_episode_length)  # 1500 by preset

        # train against classical strategy4 agents:
        # op_agent = StrategyAgent(env=env, agent_memory=2, epochs=1)
        # agent0.train(env, op_agent, strategy='always_coop', num_episodes=1)

        # get meta observation:
        meta_observation = self.env.score.reshape(1, -1)[0]  # get env score table as meta observation

        # count rewards for meta agent:
        # meta_hist = self.env.last100_history  # deque of 100 length 2 lists
        # meta_hist = [item for sublist in meta_hist for item in sublist if sublist[0] == 1 and sublist[1] == 1]  # flatten list
        # meta_reward = meta_hist.count(1)  # count number of co-op actions in game hist
        meta_reward = self.env.score[3, 1] + self.env.score[3, 0]  # get meta reward from score table

        done = False
        if (step+1) % self.low_level_episode_length == 0:
            done = True
        return meta_observation, meta_reward, done, step

    def train_by_dqn_agents(self, num_episodes):
        if torch.cuda.is_available():
            num_episodes = num_episodes
        else:
            print('No GPU found, using CPU instead.')
            num_episodes = 5

        # Note - num_steps is the maximum number of games to play between low-level agents (5000 atm)
        for i_episode in tqdm(range(num_episodes)):

            print(f'episode: {i_episode}')
            # Initialize the environment and get it's state

            if i_episode % self.meta_episode_length == 0:  # reset table only meta_episode length (2 atm)
                self.env = PdeSingleAgentEnv(memory_size=self.low_agent_memory)
                self.env.reset()  # reset table only meta_episode length
            state = self.env.score.reshape(1, -1)[0]
            dqnAgent0 = DQNAgent(self.env)  # self.episode_length = 1000
            dqnAgent1 = DQNAgent(self.env)

            state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)  # (size = 8)

            for step in count():
                ####
                # we should start agents here, as well as state. when episode is done - reset env
                ####
                action = self.select_action(state)
                observation, reward, terminated, steps_done = self.meta_step(dqnAgent0, dqnAgent1, action, step)
                reward = torch.tensor([reward], device=device)

                # find if episode is done
                done = (i_episode % self.meta_episode_length == 0)
                print('step', step, 'terminated', terminated, 'done', done)
                if terminated or done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory only in meta epoch, make sure next_state is length 200 vector
                if step % self.meta_push_pace == 0:

                    # map action back to his form in policy net
                    self.memory.push(state, action[0], next_state, reward)
                    print('push with', action[0])
                    print('action', self.map_argmax_change_to_change(action[0]),
                          self.map_argmax_cell_to_cell(action[1]))  # cell, -2, 0, 2

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                for _ in range(1):
                    self.optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                                    1 - self.TAU)
                    self.target_net.load_state_dict(target_net_state_dict)
                # if terminated or step >= self.low_level_episode_length:

                if True:
                    step = 0
                    self.episode_rewards.append(reward)
                    print(self.env.score)
                    print(f'episode reward: {reward}')

                    if (i_episode * 10 + step / dqnAgent0.episode_length) % 100 == 0:
                        self.plot_durations()
                    break

        print('Complete')
        self.plot_durations(show_result=True)
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    env = PdeSingleAgentEnv(memory_size=2)
    meta_agent = MetaDQNAgent(env)

    num_trials = 1
    social_rewards = []

    for i in range(num_trials):

        meta_agent.train_by_dqn_agents(num_episodes=1200)
    print('done')


