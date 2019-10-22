import torch.nn as nn
import torch.nn.functional as F


def init_params(layer):
    nn.init.normal_(layer.weight, 0, 0.1)
    nn.init.constant_(layer.bias, 0.1)


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(in_dim, 20)
        self.fc2 = nn.Linear(20, out_dim)
        init_params(self.fc1)
        init_params(self.fc2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(x), dim=-1)
        return out


class Critic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_dim, 20)
        self.fc2 = nn.Linear(20, out_dim)
        init_params(self.fc1)
        init_params(self.fc2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out
