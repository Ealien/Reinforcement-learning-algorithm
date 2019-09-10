import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayMemory:
    def __init__(self, n_features, memory_size=500, batch_size=32):
        self.n_features = n_features
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self.memory_counter = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample_batch(self):
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_samples = self.memory[sample_index, :]
        return batch_samples


class DeepQNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DeepQNetwork, self).__init__()
        self.l1 = nn.Linear(in_dim, 20)
        self.out = nn.Linear(20, out_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        action_value = self.out(x)
        return action_value


class DoubleDQNAgent:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.005,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 memory_size=3000,
                 batch_size=32,
                 double_q=True,
                 e_greedy_increment=None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max

        self.memory = ReplayMemory(self.n_features, memory_size, batch_size)

        self.learn_step_counter = 0

        # decide net is double dqn or not
        self.double_q = double_q

        # device: gpu or cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # network: target_net and evaluate_net
        self.eval_dqn = DeepQNetwork(self.n_features, self.n_actions).to(self.device)
        self.target_dqn = DeepQNetwork(self.n_features, self.n_actions).to(self.device)
        self.target_dqn.load_state_dict(self.eval_dqn.state_dict())
        self.target_dqn.eval()

        # optimizer
        self.optimizer = optim.RMSprop(self.eval_dqn.parameters(), lr=learning_rate)

        # loss function
        self.loss_fn = nn.MSELoss()

        self.cost_his = []

    def choose_action(self, observation):
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_dqn(observation)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_dqn.load_state_dict(self.eval_dqn.state_dict())
            print('\ntarget_parameters_replaced\n')
        samples = self.memory.sample_batch()
        loss = self.compute_dqn_loss(samples)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss)

        # increase epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def compute_dqn_loss(self, samples):
        state = torch.FloatTensor(samples[:, :self.n_features])
        action = torch.LongTensor(samples[:, self.n_features:self.n_features + 1].astype(int))
        reward = torch.FloatTensor(samples[:, self.n_features + 1:self.n_features + 2])
        state_ = torch.FloatTensor(samples[:, -self.n_features:])

        ##### Double DQN #####
        q_eval4next = self.eval_dqn(state_).detach()
        q_next = self.target_dqn(state_).detach()

        if self.double_q:
            max_eval4act = torch.argmax(q_eval4next, dim=1, keepdim=True)
            selected_q_next = q_next.gather(1, max_eval4act).view(-1, 1)
        else:
            selected_q_next = q_next.max(1)[0].view(-1, 1)

        q_target = (reward + self.gamma * selected_q_next).to(self.device)
        q_eval = self.eval_dqn(state).gather(1, action)
        # q_next = self.target_dqn(state_).detach()
        # q_target = (reward + self.gamma * q_next.max(1)[0].view(-1, 1)).to(self.device)
        loss = self.loss_fn(q_eval, q_target)
        return loss

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
