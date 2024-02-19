
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
from collections import namedtuple, deque
from itertools import count
import math
from MetaAgent import MetaAgent
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
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
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# env params:
low_agent_memory = 2
num_steps = 5000

######################################
env = PdeMetaAgentEnv(memory_size=low_agent_memory)
agent0 = Agent(env=env, agent_memory=low_agent_memory, epochs=num_steps)
agent1 = Agent(env=env, agent_memory=low_agent_memory, epochs=num_steps)
# Get number of actions from gym action space
# n_actions = env.action_space.n
n_actions = 3

# Get the number of state observations, reset gives you dict with agent_#: action.sample
obs = env.reset()
next_meta_state = env.history

n_observations = len(next_meta_state)*2  # should be 4 at start - 2 for each agent

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

memory = ReplayMemory(10000)
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    cell = (3, 0)
    if sample > eps_threshold:
        with torch.no_grad():
            state = state.reshape(1, 4)

            s = policy_net(state).max(1)[1].view(1, 1)
            return policy_net(state).max(1)[1].view(1, 1), cell
    else:
        change, cell = MetaAgent.sample_action_space()
        return torch.tensor([[change+1]], device=device, dtype=torch.long), cell

episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
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
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    #####################################

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    # print('non_final_mask', non_final_mask)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    # print('non_final_next_states', non_final_next_states)
    # print('target_net(non_final_next_states).max(1)[0].shape', target_net(non_final_next_states).max(1)[0].shape)

    # print(f'non_final_next_states{non_final_next_states}')

    ######################################
    # print('batch.state', batch.state)
    state_batch = torch.cat(batch.state)
    # print('batch.action', batch.action)
    action_batch = torch.cat(batch.action)
    # print('action_batch', action_batch)

    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print('policy_net(state_batch)', policy_net(state_batch))
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
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

# Initialize the environment and get it's state
obs = env.reset()
state0 = (obs["agent-0"], obs["agent-1"])
state1 = (obs["agent-1"], obs["agent-0"])
meta_state = env.history
done = False
cooperation_count = [0, 0]
deterministic = False
meta_epoch = 1000
meta_state = torch.tensor(meta_state[0]+meta_state[1], dtype=torch.float32, device=device).unsqueeze(0)
for step in tqdm(range(num_episodes)):
    # for t in count():
    change_action, cell = select_action(meta_state)
    meta_action = (change_action.item()-1, cell)

    action0 = agent0.predict(env, state0, deterministic)
    action1 = agent1.predict(env, state1, deterministic)

    # count cooperation times:
    if action0 == action1:
        if action0 == 1:
            cooperation_count[0] += 1
        else:
            cooperation_count[1] += 1
    actions = {"agent-0": action0, "agent-1": action1, "meta_agent": meta_action}

    next_state0, rewards, done, _ = env.step(actions)  # next = op_hist
    next_state1 = (next_state0[1], next_state0[0])  # reversed tuple
    meta_observation = env.history

    rewards["meta_agent"] = cooperation_count[0] / (cooperation_count[1]+1)

    meta_reward = torch.tensor([rewards["meta_agent"]], device=device)

    # # update meta_agent values only if it's 'epoch time'
    # if (step % meta_epoch == 0) and (step != 0):
    #     next_meta_state = env.history
    #     meta_agent.store(meta_state, meta_action, reward["meta_agent"], next_meta_state)
    #     agent0.epsilon = 0.1  # reset epsilon in-order to get new exploration period
    #     agent1.epsilon = 0.1
    #     if meta_agent.epsilon > 0.1:
    #         meta_agent.epsilon -= meta_agent.epsilon_decay
    #
    # else:
    #     meta_action = (0, (0, 0))  # reset actions for meta agent

    # fill-up personal Q-table
    agent0.store(state0, action0, rewards["agent-0"], next_state0)
    agent1.store(state1, action1, rewards["agent-1"], next_state1)

    # decay agents epsilons
    if agent0.epsilon > 0.05:
        agent0.epsilon -= agent0.epsilon_decay
        agent1.epsilon -= agent1.epsilon_decay

    if done['agent-0']:
        # print(done)
        next_state = None
    else:
        next_meta_state = torch.tensor(meta_observation[0]+meta_observation[1], dtype=torch.float32, device=device).unsqueeze(0)
        # print('next_meta_state', next_meta_state)

    # Store the transition in memory
    memory.push(meta_state, change_action, next_meta_state, meta_reward)

    # Move to the next state
    meta_state = next_meta_state

    # Perform one step of the optimization (on the policy network)
    optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)

    # if done:
    #     episode_durations.append(step + 1)
    #     # plot_durations()
    #     break
    env.write_low_level_summary(episode=step, agent0=agent0, agent1=agent1)
    env.render()

print(env.score)

# print('Complete')
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()
