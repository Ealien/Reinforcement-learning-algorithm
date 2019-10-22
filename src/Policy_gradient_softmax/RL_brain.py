import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from Policy_gradient_softmax.memory import EpisodeMemory
from Policy_gradient_softmax.model import RLNetwork


class PolicyGradient:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.95
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.memory = EpisodeMemory()

        # cpu or gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # RL Network
        self.RLNetwork = RLNetwork(self.n_features, self.n_actions).to(self.device)

        # optimizer
        self.optimizer = optim.Adam(self.RLNetwork.parameters(), lr=self.lr)

        # loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def choose_action(self, observation):
        prob_weights = self.RLNetwork(torch.FloatTensor(observation))
        prob_weights = torch.unsqueeze(prob_weights, 0).cpu().detach().numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        ep_obs = torch.FloatTensor(self.memory.sample_ep_obs()).to(self.device)
        ep_as = torch.LongTensor(self.memory.sample_ep_as()).to(self.device)
        act_eval = self.RLNetwork(ep_obs)
        loss = (self.loss_fn(act_eval, ep_as) * torch.FloatTensor(discounted_ep_rs_norm)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.clear()
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.memory.sample_ep_rs())
        running_add = 0
        for t in reversed(range(0, len(self.memory.sample_ep_rs()))):
            running_add = running_add * self.gamma + self.memory.sample_ep_rs()[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
