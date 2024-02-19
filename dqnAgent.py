import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PdeSingleAgentEnv import PdeSingleAgentEnv
from strategyagent import StrategyAgent
from agent import Agent
from torch.distributions import Categorical
from tqdm import tqdm
import os
import matplotlib.cm as cm

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):  # n_observation= memory_size, n_actions=2
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent():
    def __init__(self, env):

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

        self.memory_size = 2
        self.env = env
        self.episode_length = 10
        self. n_actions = self.env.action_space.n
        self.n_observations = 2 * self.memory_size
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(1000)
        self.episode_rewards = []
        self.steps_done = 0


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        # print(eps_threshold)
        self.steps_done += 1
        # print('state', state, 'type', type(state), 'shape', state.shape)
        if state.shape[1] <= self.memory_size:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)
        state = state.reshape(1, -1)

        softmax = nn.Softmax(dim=1)
        action_probs = softmax(self.policy_net(state))
        dist = Categorical(action_probs)
        action = dist.sample()
        return torch.tensor([[action]], device=device, dtype=torch.long)

        # if sample > eps_threshold:
        #     with torch.no_grad():
        #         # t.max(1) will return the largest column value of each row.
        #         # second column on max result is inde x of where max element was
        #         # found, so we pick action with the larger expected reward.
        #         state = state.reshape(1, -1)
        #         # print('state', state, 'type', type(state), 'shape', state.shape)
        #         return self.policy_net(state).max(1)[1].view(1, 1)
        # else:
        #     return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)

    def plot_durations(self, show_result=True):
        plt.figure(1)
        reward_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Agent0 Reward')
        plt.plot(reward_t.numpy())

        # Take 100 episode averages and plot them too
        if len(reward_t) >= 100:
            means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def plot_episode_rewards(self, op_agent, show_result=True):
        plt.figure(1)
        social_reward = (np.array(self.episode_rewards)+np.array(op_agent.episode_rewards))/2
        reward_t = torch.tensor(social_reward, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('social Reward')
        plt.plot(reward_t.numpy())

        # Take 100 episode averages and plot them too
        if len(reward_t) >= 100:
            means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
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
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        # print('batch.next_state', batch.next_state[-1].shape)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # print('batch.state', batch.state)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
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
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, env, op_agent, strategy=None, num_episodes=60):
        if torch.cuda.is_available():
            num_episodes = num_episodes
        else:
            num_episodes = 10

        if strategy is not None:
            strategy = strategy

            op_agent = StrategyAgent(env=env, agent_memory=2, epochs=self.episode_length)
            op_agent.set_strategy(strategy)

        for i_episode in range(num_episodes):
            # print(i_episode, 'i_episode')
            # Initialize the environment and get it's state
            # obs = env.reset()  # reset env
            # state = (obs["agent-0"], obs["agent-1"])
            state = [env.observation_space.sample() for _ in range(op_agent.agent_memory)]  # state len 4
            state = np.array(state).reshape(-1)
            # print('state', state, 'type', type(state))
            state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)

            for t in count():
                # print(t, 't')
                if i_episode != 0:
                    env.render()

                # state is a tuple of two arrays, agent_0_hist and agent_1_hist
                # print('state', state, 'type', type(state))

                action = self.select_action(state)

                reshaped_state = state.reshape(2, -1)
                # print('state', reshaped_state, 'type', type(reshaped_state))

                # op_state = torch.cat((reshaped_state[1], reshaped_state[0]), 0).reshape(2, -1)
                op_state = (reshaped_state[1].tolist(), reshaped_state[0].tolist())
                op_action = op_agent.predict(env, op_state)
                # print('op_action', op_action)

                actions = {"agent-0": action.item(), "agent-1": op_action}

                observation, env_reward, terminated, _ = env.step(actions)

                reward = torch.tensor([env_reward["agent-0"]], device=device)
                done = terminated['agent-0']
                if terminated['agent-0']:  # terminated is always False in this environment
                    next_state = None

                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).reshape(1,
                                                                                                                    -1)
                    # print('next_state', next_state, 'type', type(next_state))
                # Store the transition in memory
                if t > 1:  # make sure first next_state doesn't get into memory
                    self.memory.push(state, action, next_state, reward)
                    reshaped_next_state = next_state.reshape(2, -1)
                    # next_op_state = torch.cat((reshaped_next_state[1], reshaped_next_state[0]), 0).reshape(2, -1)
                    next_op_state = (reshaped_next_state[1].tolist(), reshaped_next_state[0].tolist())
                    op_agent.store(op_state, op_action, env_reward["agent-1"], next_op_state)

                # Move to the next state
                state = next_state

                if t > 1:
                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model()
                    op_agent.epsilon -= (op_agent.epsilon_decay*0.02) if op_agent.epsilon > 0.01 else 0.01
                    # print('op_agent.epsilon', op_agent.epsilon)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                                1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                self.episode_rewards.append(reward.item())

                if done or t >= self.episode_length:
                    # print('self.steps_done', self.steps_done)
                    # if op_agent.epsilon == 0.01:
                    #     op_agent.epsilon = 0.2
                    # if i_episode % 10 == 0:
                    #     print('i_episode', i_episode)
                    #     self.plot_durations()
                    # self.plot_durations()
                    break
            # print('Episode', i_episode, 'finished after', t, 'steps')

        # print('Complete')
        # self.plot_durations(show_result=True)
        plt.ioff()
        plt.show()

    def train_double_dqn(self, env, dqn_op_agent, num_episodes=60):
        if torch.cuda.is_available():
            num_episodes = num_episodes
        else:
            num_episodes = 10

        for i_episode in range(int(num_episodes/self.episode_length)):
            # Initialize the environment and get it's state
            # print('i_episode of dqn', i_episode)
            obs = env.reset()  # reset env
            state = (obs["agent-0"], obs["agent-1"])
            state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)

            for t in count():

                if i_episode != 0:
                    env.render()

                # state is a tuple of two arrays, agent_0_hist and agent_1_hist
                action = self.select_action(state)

                reshaped_state = state.reshape(2, -1)

                op_state = torch.cat((reshaped_state[1], reshaped_state[0]), 0).unsqueeze(0).reshape(1, -1)

                op_action = dqn_op_agent.select_action(op_state)

                actions = {"agent-0": action.item(), "agent-1": op_action.item()}

                observation, env_reward, terminated, _ = env.step(actions)

                reward = torch.tensor([env_reward["agent-0"]], device=device)
                op_reward = torch.tensor([env_reward["agent-1"]], device=device)

                done = terminated['agent-0']
                if terminated['agent-0']:  # terminated is always False in this environment
                    next_state = None
                    next_op_state = None

                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).reshape(1,
                                                                                                                    -1)
                    reshaped_next_state = next_state.reshape(2, -1)

                    next_op_state = torch.cat((reshaped_next_state[1], reshaped_next_state[0]), 0).unsqueeze(0).reshape(1, -1)
                    # print('next_state', next_state, type(next_state), next_state.shape)
                    # print('next_op_state', next_op_state, type(next_op_state), next_op_state.shape)


                # Store the transition in memory
                if t > 1:  # make sure first next_state doesn't get into memory
                    self.memory.push(state, action, next_state, reward)
                    dqn_op_agent.memory.push(op_state, op_action, next_op_state, op_reward)

                # Move to the next state
                state = next_state

                # control update rule for low-level agents
                if t > 1:
                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model()
                    dqn_op_agent.optimize_model()

                agents = [self, dqn_op_agent]
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                for agent in agents:
                    target_net_state_dict = agent.target_net.state_dict()
                    policy_net_state_dict = agent.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * agent.TAU + target_net_state_dict[key] * (1 - agent.TAU)
                    agent.target_net.load_state_dict(target_net_state_dict)


                # plotting durations only for agent 0
                self.episode_rewards.append(reward.item())
                dqn_op_agent.episode_rewards.append(op_reward.item())

                if done or t >= self.episode_length-1:
                    # episode_rewards.append( + 1)

                    # if i_episode % 100 == 0:
                    #     self.plot_episode_rewards(dqn_op_agent)
                        # self.plot_durations()
                    # print(t, 't of dqn')
                    break
        # self.plot_episode_rewards(dqn_op_agent)
        # print('len(self.episode_rewards)', len(self.episode_rewards))
        # print('Complete')
        # self.plot_durations(show_result=True)
        # social_reward = (np.array(self.episode_rewards)[-1000]+np.array(dqn_op_agent.episode_rewards[-1000]))/2
        # print('social_reward', social_reward.mean())
        plt.ioff()
        plt.show()
        # return social_reward.mean()

    def test_double_dqn(self, env, dqn_op_agent, num_games=1000):
        cooperation_level = []

        for i_game in range(num_games):
            # Initialize the environment and state
            obs = env.reset()
            state = (obs["agent-0"], obs["agent-1"])
            state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)

            game_rewards = []

            for t in count():
                env.render()

                # state is a tuple of two arrays, agent_0_hist and agent_1_hist
                action = self.select_action(state)

                reshaped_state = state.reshape(2, -1)
                op_state = torch.cat((reshaped_state[1], reshaped_state[0]), 0).unsqueeze(0).reshape(1, -1)
                op_action = dqn_op_agent.select_action(op_state)

                actions = {"agent-0": action.item(), "agent-1": op_action.item()}
                observation, env_reward, terminated, _ = env.step(actions)

                reward = env_reward["agent-0"]
                op_reward = env_reward["agent-1"]

                game_rewards.append((reward + op_reward) / 2)  # Calculate cooperation level for this game

                done = terminated['agent-0']
                if done or t >= self.episode_length - 1:
                    break

                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).reshape(1, -1)

            # Average the cooperation level for this game and add to list
            cooperation_level.append(np.mean(game_rewards))

        # Plot the cooperation level over games
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_games), cooperation_level)
        plt.xlabel('Game Number')
        plt.ylabel('Cooperation Level')
        plt.title('Cooperation Level over Games')
        plt.grid(True)
        plt.show()


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


def plot_cooperation_level(history, periods=200):
    # Convert tuples into cooperation levels (sum of each tuple)
    cooperation_level = [sum(game)/2 for game in history]

    # Apply moving average
    cooperation_level_smooth = moving_average(cooperation_level, periods)

    # Create an array of games (just a sequence from 1 to the length of the history)
    games = range(1, len(cooperation_level_smooth) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(games, cooperation_level_smooth, label='Cooperation Level')
    plt.xlabel('Training Progress (games played)')
    plt.ylabel('Cooperation Level')
    plt.title('Cooperation Level over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


# def plot_cooperation_levels(histories, periods=100):
#     plt.figure(figsize=(10, 6))
#
#     # Choose a color map
#     color_map = cm.get_cmap('nipy_spectral', len(histories))
#
#     # Define possible line styles
#     line_styles = ['-', '--', ':', '-.']
#
#     for i, history in enumerate(histories):
#         # Convert tuples into cooperation levels (sum of each tuple)
#         cooperation_level = [sum(game) / 2 for game in history]
#
#         # Apply moving average
#         cooperation_level_smooth = moving_average(cooperation_level, periods)
#
#         # Create an array of games (just a sequence from 1 to the length of the history)
#         games = range(1, len(cooperation_level_smooth) + 1)
#
#         # Choose line style
#         line_style = line_styles[i % len(line_styles)]
#
#         # Choose color
#         color = color_map(i)
#
#         plt.plot(games, cooperation_level_smooth, label=f'Run {i + 1}', linestyle=line_style, color=color)
#
#     plt.xlabel('Training Progress (games played)')
#     plt.ylabel('Cooperation Level')
#     plt.title('Cooperation Level over Time')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), ncol=1)
#     plt.grid(True)
#     plt.show()

def plot_cooperation_levels(histories, score_table, periods=100):
    plt.figure(figsize=(10, 6))

    # Choose a color map
    color_map = cm.get_cmap('rainbow', len(histories))

    # Define possible line styles
    line_styles = ['-', '--', ':', '-.']

    # Store final cooperation level for each line
    final_cooperation_levels = []

    for i, history in enumerate(histories):
        # Convert tuples into cooperation levels (sum of each tuple)
        cooperation_level = [sum(game) / 2 for game in history]

        # Apply moving average
        cooperation_level_smooth = moving_average(cooperation_level, periods)

        # Store final cooperation level
        final_cooperation_levels.append(cooperation_level_smooth[-1])

        # Create an array of games (just a sequence from 1 to the length of the history)
        games = range(1, len(cooperation_level_smooth) + 1)

        # Choose line style
        line_style = line_styles[i % len(line_styles)]

        # Choose color
        color = color_map(i)

        plt.plot(games, cooperation_level_smooth, label=f'Run {i + 1}', linestyle=line_style, color=color)

    plt.xlabel('Games Played')
    plt.ylabel('Cooperation Level')
    plt.ylim(0, 1)  # Set y-axis limits to be from 0 to 1
    title = "Testing Cooperation Level"  # over Time\nScore Table: " + ' '.join(map(str, score_table))
    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), ncol=1)
    plt.grid(True)
    # plt.show()

    # Return the average of the final cooperation levels
    return np.mean(final_cooperation_levels)


def plot_cooperation_levels2(histories, score_table, periods=100):
    plt.figure(figsize=(10, 6))

    # Choose a color map
    color_map = cm.get_cmap('rainbow', len(histories))

    # Define possible line styles
    line_styles = ['-', '--', ':', '-.']

    # Store final cooperation level for each line
    final_cooperation_levels = []

    for i, history in enumerate(histories):
        # Convert tuples into cooperation levels (sum of each tuple)
        cooperation_level = [sum(game) / 2 for game in history]

        # Apply moving average
        cooperation_level_smooth = moving_average(cooperation_level, periods)

        # Store final cooperation level
        final_cooperation_levels.append(cooperation_level_smooth[-1])

        # Create an array of games (just a sequence from 1 to the length of the history)
        games = range(1, len(cooperation_level_smooth) + 1)

        # Choose line style
        line_style = line_styles[i % len(line_styles)]

        # Choose color
        color = color_map(i)

        plt.plot(games, cooperation_level_smooth, label=f'Run {i + 1}', linestyle=line_style, color=color)

    plt.xlabel('Games Played')
    plt.ylabel('Cooperation Level')
    # plt.ylim(0, 1)  # Set y-axis limits to be from 0 to 1
    title = "Testing Cooperation Level"  # over Time\nScore Table: " + ' '.join(map(str, score_table))
    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), ncol=1)
    plt.grid(True)
    # plt.show()

    # Return the average of the final cooperation levels
    return np.mean(final_cooperation_levels)

def full_list():
    full_list = [[5, 5, 0, 5, 5, 0, 0, 0],
                 [4, 4, 0, 5, 5, 0, 0, 0],
                 [3, 3, 0, 5, 5, 0, 0, 0],
                 [2, 2, 0, 5, 5, 0, 0, 0],
                 [1, 1, 0, 5, 5, 0, 0, 0],
                 [0, 0, 0, 5, 5, 0, 0, 0],
                 [0, 0, 1, 5, 5, 1, 0, 0],
                 [0, 0, 2, 5, 5, 2, 0, 0],
                 [0, 0, 3, 5, 5, 3, 0, 0],
                 [0, 0, 4, 5, 5, 4, 0, 0],
                 [0, 0, 5, 5, 5, 5, 0, 0],
                 [5, 5, 0, 5, 5, 0, 1, 1],
                 [4, 4, 0, 5, 5, 0, 1, 1],
                 [3, 3, 0, 5, 5, 0, 1, 1],
                 [2, 2, 0, 5, 5, 0, 1, 1],
                 [1, 1, 0, 5, 5, 0, 1, 1],
                 [0, 0, 0, 5, 5, 0, 1, 1],
                 [0, 0, 1, 5, 5, 1, 1, 1],
                 [0, 0, 2, 5, 5, 2, 1, 1],
                 [0, 0, 3, 5, 5, 3, 1, 1],
                 [0, 0, 4, 5, 5, 4, 1, 1],
                 [0, 0, 5, 5, 5, 5, 1, 1],
                 [5, 5, 0, 5, 5, 0, 3, 3],
                 [4, 4, 0, 5, 5, 0, 3, 3],
                 [3, 3, 0, 5, 5, 0, 3, 3],
                 [2, 2, 0, 5, 5, 0, 3, 3],
                 [1, 1, 0, 5, 5, 0, 3, 3],
                 [0, 0, 0, 5, 5, 0, 3, 3],
                 [0, 0, 1, 5, 5, 1, 3, 3],
                 [0, 0, 2, 5, 5, 2, 3, 3],
                 [0, 0, 3, 5, 5, 3, 3, 3],
                 [0, 0, 4, 5, 5, 4, 3, 3],
                 [0, 0, 5, 5, 5, 5, 3, 3],
                 [5, 5, 0, 5, 5, 0, 4, 4],
                 [4, 4, 0, 5, 5, 0, 4, 4],
                 [3, 3, 0, 5, 5, 0, 4, 4],
                 [2, 2, 0, 5, 5, 0, 4, 4],
                 [1, 1, 0, 5, 5, 0, 4, 4],
                 [0, 0, 0, 5, 5, 0, 4, 4],
                 [0, 0, 1, 5, 5, 1, 4, 4],
                 [0, 0, 2, 5, 5, 2, 4, 4],
                 [0, 0, 3, 5, 5, 3, 4, 4],
                 [0, 0, 4, 5, 5, 4, 4, 4],
                 [0, 0, 5, 5, 5, 5, 4, 4],
                 [5, 5, 0, 5, 5, 0, 5, 5],
                 [4, 4, 0, 5, 5, 0, 5, 5],
                 [3, 3, 0, 5, 5, 0, 5, 5],
                 [2, 2, 0, 5, 5, 0, 5, 5],
                 [1, 1, 0, 5, 5, 0, 5, 5],
                 [0, 0, 0, 5, 5, 0, 5, 5],
                 [0, 0, 1, 5, 5, 1, 5, 5],
                 [0, 0, 2, 5, 5, 2, 5, 5],
                 [0, 0, 3, 5, 5, 3, 5, 5],
                 [0, 0, 4, 5, 5, 4, 5, 5],
                 [0, 0, 5, 5, 5, 5, 5, 5],
                 [5, 5, 0, 4, 4, 0, 5, 5],
                 [4, 4, 0, 4, 4, 0, 5, 5],
                 [3, 3, 0, 4, 4, 0, 5, 5],
                 [2, 2, 0, 4, 4, 0, 5, 5],
                 [1, 1, 0, 4, 4, 0, 5, 5],
                 [0, 0, 0, 4, 4, 0, 5, 5],
                 [0, 0, 1, 4, 4, 1, 5, 5],
                 [0, 0, 2, 4, 4, 2, 5, 5],
                 [0, 0, 3, 4, 4, 3, 5, 5],
                 [0, 0, 4, 4, 4, 4, 5, 5],
                 [0, 0, 5, 4, 4, 5, 5, 5],
                 [5, 5, 0, 3, 3, 0, 5, 5],
                 [4, 4, 0, 3, 3, 0, 5, 5],
                 [3, 3, 0, 3, 3, 0, 5, 5],
                 [2, 2, 0, 3, 3, 0, 5, 5],
                 [1, 1, 0, 3, 3, 0, 5, 5],
                 [0, 0, 0, 3, 3, 0, 5, 5],
                 [0, 0, 1, 3, 3, 1, 5, 5],
                 [0, 0, 2, 3, 3, 2, 5, 5],
                 [0, 0, 3, 3, 3, 3, 5, 5],
                 [0, 0, 4, 3, 3, 4, 5, 5],
                 [0, 0, 5, 3, 3, 5, 5, 5],
                 [5, 5, 0, 2, 2, 0, 5, 5],
                 [4, 4, 0, 2, 2, 0, 5, 5],
                 [3, 3, 0, 2, 2, 0, 5, 5],
                 [2, 2, 0, 2, 2, 0, 5, 5],
                 [1, 1, 0, 2, 2, 0, 5, 5],
                 [0, 0, 0, 2, 2, 0, 5, 5],
                 [0, 0, 1, 2, 2, 1, 5, 5],
                 [0, 0, 2, 2, 2, 2, 5, 5],
                 [0, 0, 3, 2, 2, 3, 5, 5],
                 [0, 0, 4, 2, 2, 4, 5, 5],
                 [0, 0, 5, 2, 2, 5, 5, 5],
                 [5, 5, 0, 1, 1, 0, 5, 5],
                 [4, 4, 0, 1, 1, 0, 5, 5],
                 [3, 3, 0, 1, 1, 0, 5, 5],
                 [2, 2, 0, 1, 1, 0, 5, 5],
                 [1, 1, 0, 1, 1, 0, 5, 5],
                 [0, 0, 0, 1, 1, 0, 5, 5],
                 [0, 0, 1, 1, 1, 1, 5, 5],
                 [0, 0, 2, 1, 1, 2, 5, 5],
                 [0, 0, 3, 1, 1, 3, 5, 5],
                 [0, 0, 4, 1, 1, 4, 5, 5],
                 [0, 0, 5, 1, 1, 5, 5, 5],
                 [5, 5, 0, 0, 0, 0, 5, 5],
                 [4, 4, 0, 0, 0, 0, 5, 5],
                 [3, 3, 0, 0, 0, 0, 5, 5],
                 [2, 2, 0, 0, 0, 0, 5, 5],
                 [1, 1, 0, 0, 0, 0, 5, 5],
                 [0, 0, 0, 0, 0, 0, 5, 5],
                 [0, 0, 1, 0, 0, 1, 5, 5],
                 [0, 0, 2, 0, 0, 2, 5, 5],
                 [0, 0, 3, 0, 0, 3, 5, 5],
                 [0, 0, 4, 0, 0, 4, 5, 5],
                 [0, 0, 5, 0, 0, 5, 5, 5]]
    return full_list


if __name__ == '__main__':
    # print(device)
    env = PdeSingleAgentEnv(memory_size=2)
    dqnAgent = DQNAgent(env)

    # fighting classic agents
    # op_agent = StrategyAgent(env=env, agent_memory=2, epochs=1000)
    # dqnAgent.train(env, op_agent, strategy='tit_for_tat', num_episodes=1000)

    # fighting q learning agents
    # op_agent = Agent(env=env, agent_memory=2, epochs=1000)
    # dqnAgent.train(env, op_agent, num_episodes=25)

    # # fighting dqn agents
    dqn_op_agent = DQNAgent(env)
    # dqnAgent.train_double_dqn(env, dqn_op_agent, num_episodes=130)

    full_list = full_list()
    cp_levels = []
    num_trials = 20
    social_rewards = []
    histories = []
    for ls in full_list:
        for i in tqdm(range(num_trials)):
            env = PdeSingleAgentEnv(memory_size=2)
            env.score = np.array(ls).reshape(4, 2)
            dqnAgent = DQNAgent(env)

            dqn_op_agent = DQNAgent(env)

        ## plot block ##
            ## print social reward
            social_reward = dqnAgent.train_double_dqn(env, dqn_op_agent, num_episodes=1500)
            social_rewards.append(social_reward)

            ## plot cooperation level
            history = env.total_history
            ##plot_cooperation_level(history)
            histories.append(history)

        avg_cooperation = plot_cooperation_levels(histories, env.score.reshape(8, -1))
        cp_levels.append(avg_cooperation)
    print('cp_levels', cp_levels)
    # print('social_rewards', social_rewards)
## end here ##


### IMPORTANT NOTE:
# In order to go back to full run (with meta), remvoe the following lines:
# in if main==__main__: social rewards and histories. plot_c_level too.

# in train_double_dqn: social reward