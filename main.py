from collections import deque
from gym.spaces import Discrete, Box
import numpy as np
from gym_example import Pde
from agent import Agent
from strategyagent import StrategyAgent
from Qlearning import QLearning
from PdeSingleAgentEnv import PdeSingleAgentEnv
import time
from MetaAgent import MetaAgent
from PdeMetaAgentEnv import PdeMetaAgentEnv
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import seaborn as sns
from plotter import Plotter


def image_summary(imaged_plots):
    # Sets up a timestamped log directory.
    logdir = "runs/images/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    figure = Plotter.image_grid(imaged_plots)
    with file_writer.as_default():
      tf.summary.image("Agents Heatmaps", Plotter.plot_to_image(figure), step=0)


def init_q_table():
    q_table0 = np.array([[261.57420374, 288.1765307],
                [267.2416887,  205.39825126],
                [284.60594318, 254.80881419],
                [275.77106382, 193.83766445],
                [257.82066416, 174.43457588],
                [264.22382873, 288.20650068],
                [269.88672624, 168.71709484],
                [208.41414867, 252.1874414],
                [275.91830996, 251.0536695],
                [275.19478756, 174.53098721],
                [266.68833999, 226.33575093],
                [279.70808776, 238.54172333],
                [253.16552863, 138.91001759],
                [230.11461678, 203.47308032],
                [277.44463026, 201.03313548],
                [279.24763613, 290.66357943]])
    return q_table0


def init_converged_q_table():
    q_table0 = np.array([[170.80621752, 283.19938491],
                [269.90126673, 175.30243875],
                [202.08815835, 283.25811379],
                [212.63694222, 188.61571399],
                [286.26139261, 154.04831525],
                [275.26171195, 293.08182118],
                [288.64139591, 179.02814965],
                [190.3447343,  227.46548433],
                [246.75437904, 292.73755815],
                [274.88165414, 149.98930793],
                [281.73354792, 170.31417297],
                [208.99810585, 284.06744114],
                [280.14549553, 128.96556992],
                [178.3071865,  173.28390449],
                [288.72362063, 242.08667116],
                [285.27517816,  300.]])
    return q_table0


def gym_example_run():
    num_steps = 100000
    # game agents and memory size for each agent
    memory_size = 2
    env = Pde(memory_size=memory_size)
    agent0 = QLearning(env=env, agent_memory=memory_size)
    agent1 = QLearning(env=env, agent_memory=memory_size)
    learn_every = 1

    # Number of steps you run the agent for
    model = QLearning(env)
    q_table0, q_table1, cooperation_count = model.learn(env, num_steps, agent0, agent1, memory_size)
    # print(f' q_table0:, {q_table0}')
    # print(f' q_table1:, {q_table1}')
    print(f' env history:{env.history}')
    print(f'Cooperation times {cooperation_count[0]}'
          f' double-defection times {cooperation_count[1]} out of {num_steps} epochs')

    return q_table0, q_table1, cooperation_count


def double_agent_run(alpha):
    num_steps = 5000
    memory_size = 2
    env = PdeSingleAgentEnv(memory_size=memory_size)
    agent0 = Agent(env=env, agent_memory=memory_size, epochs=num_steps)
    agent0.alpha = alpha
    # agent0.q_table = init_converged_q_table()
    agent1 = Agent(env=env, agent_memory=memory_size, epochs=num_steps)
    agent1.alpha = alpha
    # agent1.q_table = init_converged_q_table()
    agent0.q_table, agent1.q_table, cooperation_count = Agent.learn_as_single_agent(env, num_steps, agent0, agent1)
    print(f' env history:{env.history}')
    print(f' cooperation level {PdeSingleAgentEnv.compute_cooperation_level(env.history)}')
    print(f'Cooperation times {cooperation_count[0]}'
          f' double-defection times {cooperation_count[1]} out of {num_steps} epochs')
    return agent0.q_table, agent1.q_table, cooperation_count


def single_agent_run(strategy):
    num_steps = 5000
    memory_size = 2
    env = PdeSingleAgentEnv(memory_size=memory_size)
    agent0 = Agent(env=env, agent_memory=memory_size, epochs=num_steps)
    # agent0.q_table = init_converged_q_table()
    agent1 = StrategyAgent(env=env, agent_memory=memory_size, epochs=num_steps)
    agent1.set_strategy(strategy)
    agent0.q_table, agent1.q_table, cooperation_count = Agent.learn_as_single_agent(env, num_steps, agent0, agent1)
    print(f' env history:{env.history}')
    print(f' cooperation level {PdeSingleAgentEnv.compute_cooperation_level(env.history)}')
    print(f'Cooperation times {cooperation_count[0]}'
          f' double-defection times {cooperation_count[1]} out of {num_steps} epochs')
    return agent0.q_table, agent1.q_table, cooperation_count


def meta_agent_run():
    num_steps = 10000
    memory_size = 2
    env = PdeMetaAgentEnv(memory_size=memory_size)
    agent0 = Agent(env=env, agent_memory=memory_size, epochs=num_steps)
    agent1 = Agent(env=env, agent_memory=memory_size, epochs=num_steps)
    meta_agent = MetaAgent(env=env, agent_memory=memory_size, epochs=num_steps)
    agent0.q_table, agent1.q_table, meta_agent.meta_q_table, cooperation_count =\
        MetaAgent.learn_meta_agent(env, num_steps, agent0, agent1, meta_agent)
    print(f' Last env history:{env.history}')
    print(f' Co-operation level {PdeSingleAgentEnv.compute_cooperation_level(env.history)}')
    print(f'Co-operation times {cooperation_count[0]}'
          f' Double-defection times {cooperation_count[1]} out of {num_steps} epochs')
    print(f'Env ending score table:{env.score}')
    return agent0.q_table, agent1.q_table, meta_agent.meta_q_table, cooperation_count


def meta_strategy_agent_run(strategy):
    num_steps = 10000
    memory_size = 2
    env = PdeMetaAgentEnv(memory_size=memory_size)
    agent0 = Agent(env=env, agent_memory=memory_size, epochs=num_steps)
    agent1 = StrategyAgent(env=env, agent_memory=memory_size, epochs=num_steps)
    agent1.set_strategy(strategy)
    meta_agent = MetaAgent(env=env, agent_memory=memory_size, epochs=num_steps)

    agent0.q_table, agent1.q_table, meta_agent.meta_q_table, cooperation_count = \
        MetaAgent.learn_meta_agent(env, num_steps, agent0, agent1, meta_agent)
    print(f' Last env history:{env.history}')
    print(f' Co-operation level {PdeSingleAgentEnv.compute_cooperation_level(env.history)}')
    print(f'Co-operation times {cooperation_count[0]}'
          f' Double-defection times {cooperation_count[1]} out of {num_steps} epochs')
    print(f'Env ending score table:{env.score}')
    return agent0.q_table, agent1.q_table, meta_agent.meta_q_table, cooperation_count


if __name__ == '__main__':
    start_time = time.time()

    tf.config.experimental.set_visible_devices([], 'GPU')
    #  q_table0, q_table1, cooperation_count = gym_example_run()
    strategy = 'tit_for_tat'
    q_table0, q_table1, cooperation_count = single_agent_run(strategy)
    # q_table0, q_table1, cooperation_count = double_agent_run(alpha=0.3)
    # q_table0, q_table1, meta_q_table, cooperation_count = meta_agent_run()
    # q_table0, q_table1, coop = double_agent_run()

    # num_of_runs = 3
    # q_table0, q_table1, meta_q_table, cooperation_count = meta_strategy_agent_run(strategy)
    # q_table0, q_table1, meta_q_table, cooperation_count = meta_agent_run()

    print(f' q_table0 = {q_table0}')
    print(f' q_table0 = {q_table0.shape}')

    q_tables = [q_table0, q_table1, np.absolute(q_table0-q_table1)]
    # q_tables = [q_table0, q_table1, np.absolute(q_table0-q_table1), meta_q_table]

    imaged_plots = {'class_name': ['q0', 'q1', 'q1-q0'], 'images': q_tables}
    # imaged_plots = {'class_name': ['q0', 'q1', 'q1-q0', 'Meta_q_table'], 'images': q_tables}


    # plot heatmaps of q_tables
    image_summary(imaged_plots)
    print("--- %s seconds ---" % (time.time() - start_time))

    # print(f' meta_table = {meta_q_table}')
    # print(f' meta_table.shape = {meta_q_table.shape}')
