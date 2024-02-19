from gym_example import Pde
from agent import Agent
from Qlearning import QLearning
"""
env.play([int1, int2])
int1 - agent 1 action
int2 - agent 2 action
[r1, r2]

memory = 2
obs['agent-0'] = [0, 1]
"""

# game agents and memory size for each agent
memory_size = 2
env = Pde(memory_size=memory_size)
agent0 = QLearning(env=env, agent_memory=memory_size)
agent1 = QLearning(env=env, agent_memory=memory_size)
learn_every = 1

# Number of steps you run the agent for
num_steps = 1500
def learn (self, env, num_steps):
    # reset env
    obs = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False

    # in our case num_steps = num epochs
    for step in range(num_steps):
        # if action there is not enough actions_history, make a random action
        if step <= memory_size:
            action0 = env.action_space.sample()
            action1 = env.action_space.sample()

        else:
            action0 = agent0.predict(obs["agent-0"])
            action1 = agent1.predict(obs["agent-1"])
        actions = {"agent-0": action0, "agent-1": action1}

        # apply the action and interact with the environment
        obs, reward, done, _ = env.step(actions)

        # fill-up personal Q-table
        agent0.store(obs, reward, action0)
        agent1.store(obs, reward, action1)
        """
        if step % learn_every == 0:
            agent0.learn(env)
            agent1.learn(env)
        """

        # Render the env
        env.render()

        # no need for env reset in our implementation
        # # If the episode is up, then start another one
        # if done:
        #     env.reset()
    return agent0.q_table, agent1.q_table



    # agent should learn as function of - state, reward, action
    def learn(self, env):
        if env.env_name == "PDE":
            self.q_table = self.q_Table(env)  # ?

            for i in range(self.epochs):
                state = env.reset()  # just making Done = false

                epochs, penalties, reward, = 0, 0, 0
                done = False

                while not done:
                    if random.uniform(0, 1) < self.epsilon:
                        action = env.action_space.sample()  # Explore action space

                        # # forbidden illegal actions:
                        # while state[int(action / 2)][action % 2] != 0:
                        #     action = env.action_space.sample()

                    else:
                        # assign to q_table[state,action] where state is a vector the size of memory
                        action_value_list = self.q_table[self.state_to_row(state)]  # take table with specific row

                        for action, action_value in enumerate(action_value_list):
                            if action_value == 0:
                                action_value_list[action] = np.nan
                        #### set zeroes to np.nan??????? ####

                        action = np.nanargmax(action_value_list)  # Exploit learned values
                    next_state, reward, done, info = env.step(action)

                    old_value = self.q_table[self.state_to_number(state), action]
                    next_max = np.nanmax(self.q_table[self.state_to_number(next_state)])
                    new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                    self.q_table[self.state_to_number(state), action] = new_value
                    state = next_state

                    epochs += 1
        return self.q_table