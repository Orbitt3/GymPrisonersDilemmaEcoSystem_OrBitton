import torch
import torch.nn as nn
import torch.optim as optim
from dqnAgent import DQNAgent
from PdeVecEnv import PdeVecEnv
from PdeSingleAgentEnv import PdeSingleAgentEnv
from gym.spaces import Discrete, Box


class CooperationPredictor(nn.Module):
    def __init__(self):
        super(CooperationPredictor, self).__init__()
        # Define model
        self.input_dim = 8
        self.hidden_dim = 32
        self.output_dim = 1

        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.relu = nn.ReLU()



        # Define loss
        self.criterion = nn.MSELoss()

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.1)

    def forward(self, x):
        x = x.float()
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

    # Training loop
    def train_predictor(self, x, y, epochs):
        self.train()
        y=y.float()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

    # Predict new samples
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)
        return outputs

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    @staticmethod
    def get_reward(low_env):
        # count rewards for meta agent:
        meta_hist = low_env.last100_history  # deque of 100 length 2 lists
        meta_hist = [item for sublist in meta_hist for item in sublist if sublist[0] == 1 and sublist[1] == 1]  # flatten list
        meta_reward = meta_hist.count(1)  # count number of co-op actions in game hist
        return meta_reward

    def collect_data(self, num_episodes=500, low_level_episode_length=1000):
        # Collect data from the environment
        X = []
        Y = []
        for episode in range(num_episodes):
            # init agents
            print('episode:', episode)
            low_env = PdeSingleAgentEnv(memory_size=2)
            low_env.score_table = torch.randint(-6, 6, (8,))
            print('score_table:', low_env.score_table, low_env.score_table.shape)
            dqnAgent0 = DQNAgent(low_env)  # self.episode_length = 1000
            dqnAgent1 = DQNAgent(low_env)
            dqnAgent0.train_double_dqn(low_env, dqnAgent1, num_episodes=low_level_episode_length)

            reward = self.get_reward(low_env)

            X.append(low_env.score_table.unsqueeze(0))  # Add extra dimension to tensor
            Y.append(torch.tensor([reward]))  # Make reward a tensor
        return torch.cat(X, 0), torch.cat(Y, 0)  # Convert lists of tensors to single tensor

if __name__ == '__main__':
    model = CooperationPredictor()

    # Suppose we have training data X and targets Y
    # X = torch.randn(100, 8)  # 1000 input vectors
    # Y = torch.randint(0, 201, (100,)).float()  # 1000 target integers

    X, Y = model.collect_data(num_episodes=2500, low_level_episode_length=1500)

    # Convert X and Y to tensors
    print('X:', X, 'Y:', Y, type(X), type(Y))
    # X_tensor = torch.FloatTensor(X)
    # Y_tensor = torch.FloatTensor(Y)

    # Train the model
    model.train_predictor(X, Y, epochs=500)

    # X = torch.randn(1, 8)  # 1000 input vectors
    X = [1, 1, 0, 2, 2, 0, 0, 0]
    X_tensor = torch.FloatTensor(X)
    print(X_tensor)
    print(model.predict(X_tensor))

    X = [1, 1, 0, 5, 5, 0, 3, 3]
    X_tensor = torch.FloatTensor(X)
    print(X_tensor)
    print(model.predict(X_tensor))

    X = [1, 1, 0, 2, 2, 0, 3, 3]
    X_tensor = torch.FloatTensor(X)
    print(X_tensor)
    print(model.predict(X_tensor))
