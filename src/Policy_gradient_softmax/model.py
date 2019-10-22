import torch
import torch.nn as nn
import torch.nn.functional as F


class RLNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RLNetwork, self).__init__()
        self.fc1 = nn.Linear(in_dim, 10)
        self.fc2 = nn.Linear(10, out_dim)
        self.init_params(self.fc1)
        self.init_params(self.fc2)

    def init_params(self, layer):
        nn.init.normal_(layer.weight, 0, 0.3)
        nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        out = torch.tanh(self.fc1(x))
        out = self.fc2(out)
        out = F.softmax(out, dim=-1)
        return out
