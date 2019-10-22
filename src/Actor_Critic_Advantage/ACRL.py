import numpy as np
import torch
import torch.optim as optim

from Actor_Critic_Advantage.model import Actor, Critic


class ACTrainer:
    def __init__(self, n_actions, n_features, lr_a=0.001, lr_c=0.01, gamma=0.9):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma

        # Device: cpu or gpu
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self.actor = Actor(self.n_features, self.n_actions).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr_a)

        # Critic
        self.critic = Critic(self.n_features, 1).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, s):
        # s = s[np.newaxis, :]
        prob = self.actor(torch.FloatTensor(s))
        prob = torch.unsqueeze(prob, 0).cpu().detach().numpy()
        action = np.random.choice(range(prob.shape[1]), p=prob.ravel())
        return action

    def update(self, s, s_, a, r):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v = self.critic(torch.FloatTensor(s))
        v_ = self.critic(torch.FloatTensor(s_))

        # Critic loss
        td_error = r + self.gamma * v_ - v
        critic_loss = torch.pow(td_error, 2)

        # Critic update
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        td_error = td_error.detach()
        acts_prob = self.actor(torch.FloatTensor(s))
        log_prob = torch.log(acts_prob)
        exp_v = torch.mean(log_prob * td_error)
        actor_loss = -exp_v

        # Actor update
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
