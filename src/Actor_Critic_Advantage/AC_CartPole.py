import gym
import numpy as np

from Actor_Critic_Advantage.ACRL import ACTrainer

np.random.seed(2)

# Superparameters
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible, general Policy gradient has high variance
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

ac_trainer = ACTrainer(
    n_actions=N_A,
    n_features=N_F,
    lr_a=LR_A,
    lr_c=LR_C,
    gamma=GAMMA
)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = ac_trainer.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        ac_trainer.update(s, s_, a, r)

        s = s_

        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
