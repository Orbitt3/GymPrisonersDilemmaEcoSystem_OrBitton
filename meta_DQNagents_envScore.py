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
        self.layer1 = nn.Linear(n_observations, 16)
        # self.conv1 = nn.Conv2d(n_observations, 4, (3, 3), padding=1, stride=1)
        # self.conv2 = nn.Conv2d(4, 4, (4, 2), stride=(2, 1))
        # self.fc = nn.Linear(84, n_actions)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # batch_size = x.shape[0]
        x = F.relu((self.layer1(x)))
        x = F.relu((self.layer2(x)))
        # x = x.view(batch_size, -1)
        return self.layer3(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TAU = 0.005
LR = 1e-2  # 0.1

# env params:
low_agent_memory = 2
num_steps = 5000
episode_rewards = []
######################################
env = PdeMetaAgentEnv(memory_size=low_agent_memory)

# Settling two DQN agents
dqnAgent0 = DQNAgent(env)
dqnAgent1 = DQNAgent(env)

# Settling two q-learning agents
# agent0 = Agent(env=env, agent_memory=low_agent_memory, epochs=num_steps)
# agent1 = Agent(env=env, agent_memory=low_agent_memory, epochs=num_steps)


# Get number of actions from gym action space
# n_actions = env.action_space.n
n_actions = 3

# Get the number of state observations, reset gives you dict with agent_#: action.sample
obs = env.reset()

n_observations = 8  # env.score table

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

memory = ReplayMemory(10000)
steps_done = 0


def select_action(state, episode_done):
    # print('state.shape', state.shape, 'state', state)

    global steps_done
    sample = random.random()
    # eps_threshold = EPS_START - ((EPS_START - EPS_END) / (num_episodes-25)) * episode_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (episode_done) / EPS_DECAY)
    steps_done += 1
    # print('steps_done', steps_done)
    cell = (3, 0)
    # print(eps_threshold)

    softmax = nn.Softmax(dim=1)
    action_probs = softmax(policy_net(state))
    dist = Categorical(action_probs)
    change = dist.sample()
    return torch.tensor([[change-1]], device=device, dtype=torch.long), cell

    # if sample > eps_threshold and steps_done > n_observations:
    #     with torch.no_grad():
    #         # t.max(1) will return the largest column value of each row.
    #         # second column on max result is index of where max element was
    #         # found, so we pick action with the larger expected reward.
    #         # s = policy_net(state).max(1)[1].view(1, 1)
    #         return policy_net(state).max(1)[1].view(1, 1)-1, cell  # -1 to get -1,0,1
    # else:
    #     change, cell = MetaAgent.sample_action_space()
    #     return torch.tensor([[change]], device=device, dtype=torch.long), cell


def plot_durations(show_result=False):
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
    if len(reward_steps) >= 10:
        means = reward_steps.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    # plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
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

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    # print('optimized')

    optimizer.step()


def meta_step(env, agent0, agent1, action, step):
    # low-level episode length:
    max_episode_length = 100

    # score table bounds:
    bounds = [-10, 10]

    change, cell = action
    change *= 1  # change can be -1, 0, 1, but from policy net it is 0, 1, 2
    # let meta update score table only in 100th step
    if step % meta_push_pace != 0:
        change *= 0

    # check bounds of score table
    new_val = env.score[3, 1] + change.item()*1
    if new_val <= bounds[1] and new_val >= bounds[0]:
        env.score[cell[0], cell[1]] += change.item()*1  # score_Table cell += meta_change
        env.score[3, 1] += change.item()*1

    # let two dqn_agents play with new environment:
    # agent0.train_double_dqn(env, agent1, num_episodes=1)

    # train against classical strategy agents:
    op_agent = StrategyAgent(env=env, agent_memory=2, epochs=1)
    agent0.train(env, op_agent, strategy='tit_for_tat', num_episodes=1)

    # get meta observation:
    meta_observation = env.score.reshape(1, -1)[0]  # get env score table as meta observation

    # count rewards for meta agent:
    meta_hist = env.last100_history  # deque of 100 length 2 lists
    meta_hist = [item for sublist in meta_hist for item in sublist]  # flatten list
    meta_reward = meta_hist.count(1)  # count number of co-op actions in game hist
    # meta_reward = env.score[3, 1] + env.score[3, 0]  # get meta reward from score table

    # check if episode is done:
    done = False
    if (step+1) % max_episode_length == 0:
        done = True

    return meta_observation, meta_reward, done, step


if torch.cuda.is_available():
    num_episodes = 400
else:
    num_episodes = 50

# Meta hyper-parameters:
low_agent_epoch = 1000
low_agents_games_per_epoch = 10  # (t)
meta_push_pace = 1
plot_pace = 1  # int(num_episodes / 12)

# explanation for epochs:
#  1 episode = 1100 training steps
#  1 dqn epoch = 10 steps


# Note - num_steps is the maximum number of games to play between low-level agents (5000 atm)
for i_episode in range(num_episodes):
    print(f'episode: {i_episode}')
    # Initialize the environment and get it's state
    env = PdeSingleAgentEnv(memory_size=low_agent_memory)
    state = env.reset()
    state = env.score.reshape(1, -1)[0]
    dqnAgent0 = DQNAgent(env)  # self.episode_length = 1000
    dqnAgent1 = DQNAgent(env)

    state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)  # (size = 8)

    # else:
    #     state = env.last100_history
    #     print('do not change anything')

    for step in count():
        # print('step', step)
        action = select_action(state, i_episode)
        # print('action', action)

        observation, reward, terminated, steps_done = meta_step(env, dqnAgent0, dqnAgent1, action, step)
        # print('action', action)

        reward = torch.tensor([reward], device=device)

        # print('terminated', terminated)
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory only in meta epoch, make sure next_state is length 200 vector
        if step % meta_push_pace == 0 and len(env.last100_history) >= 100:
            # print('step', step)
            memory.push(state, action[0]+1, next_state, reward)
            # print('global_steps_done:', dqnAgent0.steps_done)
            # print(state, action[0]+1, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        # if terminated or we got to maximum number of steps
        if terminated or step >= low_agent_epoch:
            step = 0
            episode_rewards.append(reward)
            eps_threshold = EPS_START - ((EPS_START - EPS_END) / num_episodes) * i_episode
            # print('eps_threshold:', eps_threshold)
            print('episode_reward:', reward)
            print(env.score)

            if (i_episode*10+step/dqnAgent0.episode_length) % plot_pace == 0:
                plot_durations()
                # print(eps_threshold)
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
