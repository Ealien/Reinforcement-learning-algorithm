import gym
import matplotlib.pyplot as plt
import numpy as np

from Double_DQN.RL_brain_torch import DoubleDQNAgent

# from RL_brain_torch import DoubleDQNAgent

env = gym.make('Pendulum-v0')
ACTION_SPACE = 11
MEMORY_SIZE = 3000

natural_DQN = DoubleDQNAgent(
    n_actions=ACTION_SPACE,
    n_features=3,
    e_greedy_increment=0.001,
    double_q=False
)

double_DQN = DoubleDQNAgent(
    n_actions=ACTION_SPACE,
    n_features=3,
    e_greedy_increment=0.001,
    double_q=True
)


def train(RL):
    total_step = 0

    observation = env.reset()

    while True:
        print('total_step: ', total_step, '\n')
        if total_step - MEMORY_SIZE > 8000:
            env.render()

        action = RL.choose_action(observation)

        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10
        RL.memory.store_transition(observation, action, reward, observation_)

        if total_step > MEMORY_SIZE:
            RL.learn()

        if total_step - MEMORY_SIZE > 15000:
            break
        observation = observation_
        total_step += 1
    print('game over')
    env.close()
    return RL.q


if __name__ == '__main__':
    q_natural = train(natural_DQN)
    q_double = train(double_DQN)

    plt.plot(np.array(q_natural), c='r', label='natural')
    plt.plot(np.array(q_double), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()
