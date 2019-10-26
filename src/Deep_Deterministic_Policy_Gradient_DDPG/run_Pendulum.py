import gym
import numpy as np

from Deep_Deterministic_Policy_Gradient_DDPG.ddpg import DDPG

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
print(s_dim)
print(a_dim)
print(a_bound)

ddpg = DDPG(
    s_dim=s_dim,
    a_dim=a_dim,
    a_bound=a_bound,
    reward_decay=GAMMA,
    lr_a=LR_A,
    lr_c=LR_C,
    memory_size=MEMORY_CAPACITY,
    batch_size=BATCH_SIZE,
    tau=TAU
)

var = 3

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)

        s_, r, done, info = env.step(a)

        ddpg.memory.store_transitions(s, s_, r / 10, a)

        if ddpg.memory.memory_count > 10000:
            var *= .9995  # decay the action randomness
            ddpg.update()

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:
                RENDER = True
            break
