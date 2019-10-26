import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super(Actor, self).__init__()
        self.a_bound = torch.FloatTensor(a_bound)
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc2 = nn.Linear(30, a_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.mul(x, self.a_bound)


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, out_dim):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(s_dim, 30)
        self.fc_a = nn.Linear(a_dim, 30)
        self.fc2 = nn.Linear(30, out_dim)

    def forward(self, s, a):
        out = F.relu(self.fc_s(s) + self.fc_a(a))
        out = self.fc2(out)
        return out
