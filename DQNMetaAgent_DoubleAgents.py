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
        # self.layer1 = nn.Linear(n_observations, 128)
        self.conv1 = nn.Conv2d(n_observations, 4, (3, 3), padding=1, stride=1)
        self.conv2 = nn.Conv2d(4, 4, (4, 2), stride=(2, 1))
        self.fc = nn.Linear(84, n_actions)


        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        batch_size = x.shape[0]
        # print('x.shape', x.shape)
        # x = x.reshape(1, x.shape[0], x.shape[1])
        # x = x.unsqueeze(0)
        # x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))

        x = F.relu((self.conv1(x)))

        x = F.relu((self.conv2(x)))
        # x = torch.flatten(x, 1)
        x = x.view(batch_size, -1)
        # print('x.shape', x.shape)

        # return self.layer3(x)
        return self.fc(x)



# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer


BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# env params:
low_agent_memory = 2
num_steps = 5000
episode_rewards = []
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

# n_observations = len(next_meta_state) * 2  # should be 4 at start - 2 for each agent
# n_observations = 32  # should be 4 at start - 2 for each agent
n_observations = 1  # one 'image'

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

memory = ReplayMemory(1000)
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    cell = (3, 0)
    print(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # state = state.reshape(1, 32)

            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # s = policy_net(state).max(1)[1].view(1, 1)
            return policy_net(state).max(1)[1].view(1, 1), cell
    else:
        change, cell = MetaAgent.sample_action_space()
        return torch.tensor([[change + 1]], device=device, dtype=torch.long), cell


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
    if len(reward_steps) >= 100:
        means = reward_steps.unfold(0, 100, 1).mean(1).view(-1)
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
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print('action_batch.shape', action_batch.shape)
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
    print('optimized')

    optimizer.step()


def meta_step(env, agent0, agent1, action, num_steps, step):
    # reduce epoch duration to get small transitions
    steps_made = 100*step  # every step is meta-transition
    num_steps -= steps_made

    # update score table
    change, cell = action
    change *= 1
    env.score[cell[0], cell[1]] += change.item()*10  # score_Table cell += meta_change
    env.score[3, 1] += change.item()*10

    # let two agents play with new environment
    agent0.q_table, agent1.q_table, cooperation_count = Agent.learn_as_single_agent(env, num_steps, agent0, agent1)

    # unpack variables:
    # meta_observation = np.concatenate((agent0.q_table[:, 1], agent1.q_table[:, 1]))
    meta_observation = np.concatenate((agent0.q_table, agent1.q_table), axis=1)
    # print(f'meta_observation{meta_observation.shape}')
    meta_reward = cooperation_count[0]  #/(cooperation_count[1] + 1)
    info = (agent0.q_table, agent1.q_table)
    done = False
    if num_steps < 100:
        done = True
    return meta_observation, meta_reward, done, info


if torch.cuda.is_available():
    num_episodes = 200
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    print(f'i_episode: {i_episode}')
    # Initialize the environment and get it's state
    env = PdeSingleAgentEnv(memory_size=low_agent_memory)
    state, _ = env.reset()
    agent0 = Agent(env=env, agent_memory=low_agent_memory, epochs=num_steps)
    agent1 = Agent(env=env, agent_memory=low_agent_memory, epochs=num_steps)
    # agent1 = StrategyAgent(env=env, agent_memory=low_agent_memory, epochs=num_steps)
    # agent1.set_strategy('always_defect')


    # set state for meta agent
    # state = np.concatenate((agent0.q_table[:, 1], agent1.q_table[:, 1]))
    state = np.concatenate((agent0.q_table, agent1.q_table), axis=1)
    # print(f'state.shape{state.reshape(1,-1).shape}')
    # print(f'[state].shape{[state].shape}')

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1x16x4)
    # print('state.shape', state.shape)

    for step in count():
        action = select_action(state)

        observation, reward, terminated, info = meta_step(env, agent0, agent1, action, num_steps, step)

        agent0.q_table, agent1.q_table = info

        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None
        else:
            print('next_state_entered')
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action[0], next_state, reward)
        print(f'reward {reward}')
        print(f'action[0] {action[0]}')


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

        # done = True  # works if we have one actions per epoch
        if terminated:
            episode_rewards.append(reward)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
