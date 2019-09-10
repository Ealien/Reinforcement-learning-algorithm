import gym
import matplotlib.pyplot as plt
from Double_DQN.RL_brain_torch import DoubleDQNAgent
import numpy as np

env = gym.make('Pendulum-v0')
ACTION_SPACE = 11
MEMORY_SIZE = 3000

natural_DQN = DoubleDQNAgent(
    n_actions=ACTION_SPACE,
    n_features=3,
    e_greedy_increment=0.001,
    double_q=True
)


def train(RL):
    total_step = 0

    observation = env.reset()

    while True:
        if total_step - MEMORY_SIZE > 8000:
            env.render()

        action = RL.choose_action(observation)

        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10
        RL.memory.store_transition(observation, action, reward, observation_)

        if total_step > MEMORY_SIZE:
            RL.learn()

        if total_step - MEMORY_SIZE > 20000:
            break
        observation = observation_
        total_step += 1


if __name__ == '__main__':
    train(natural_DQN)
