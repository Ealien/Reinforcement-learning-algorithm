import torch
import torch.nn as nn
import torch.optim as optim

from Deep_Deterministic_Policy_Gradient_DDPG.memory import Memory
from Deep_Deterministic_Policy_Gradient_DDPG.model import Actor, Critic


def soft_replacement(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


class DDPG:
    def __init__(self,
                 s_dim,
                 a_dim,
                 a_bound,
                 reward_decay=0.9,
                 lr_a=0.001,
                 lr_c=0.002,
                 memory_size=5000,
                 batch_size=32,
                 tau=0.01):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.gamma = reward_decay
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.train_replace_iter = 300

        self.memory = Memory(s_dim, a_dim, memory_size, batch_size)

        # device: cpu or gpu
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        # Actor Network
        self.eval_a = Actor(s_dim, a_dim, a_bound).to(self.device)
        self.target_a_ = Actor(s_dim, a_dim, a_bound).to(self.device)

        # Critic Network
        self.eval_q = Critic(self.s_dim, self.a_dim, 1).to(self.device)
        self.target_q_ = Critic(self.s_dim, self.a_dim, 1).to(self.device)

        # hard replacement
        hard_update(self.target_a_, self.eval_a)
        hard_update(self.target_q_, self.eval_q)
        self.target_a_.eval()
        self.target_q_.eval()

        # Actor loss_fn and optimizer
        self.actor_optimizer = optim.Adam(self.eval_a.parameters(), lr=self.lr_a)

        # Critic loss_fn and optimizer
        self.critic_optimizer = optim.Adam(self.eval_q.parameters(), lr=self.lr_c)
        self.critic_loss = nn.MSELoss()

        self.train_step = 0

    def choose_action(self, s):
        action = self.eval_a(torch.FloatTensor(s))
        return action.cpu().detach().numpy()

    def update(self):
        # hard replacement
        # if self.train_step % self.train_replace_iter == 0:
        #     hard_update(self.target_a_, self.eval_a)
        #     hard_update(self.target_q_, self.eval_q)

        bt = torch.FloatTensor(self.memory.sample_batch()).to(self.device)
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # critic update
        a_ = self.target_a_(bs_)
        q_ = self.target_q_(bs_, a_)
        target_q = (br + self.gamma * q_).detach()
        q = self.eval_q(bs, ba)
        critic_loss = self.critic_loss(q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        a = self.eval_a(bs)
        actor_loss = -torch.mean(self.eval_q(bs, a))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_replacement(self.target_q_, self.eval_q, self.tau)
        soft_replacement(self.target_a_, self.eval_a, self.tau)

        self.train_step += 1
