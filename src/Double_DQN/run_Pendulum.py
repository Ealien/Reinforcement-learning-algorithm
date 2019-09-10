import gym
import numpy as np
import tensorflow as tf

from Double_DQN.RL_brain import DoubleDQN

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
ACTION_SPACE = 11
MEMORY_SIZE = 3000

sess = tf.Session()

with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE,
        n_features=3,
        memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001,
        double_q=False,
        sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE,
        n_features=3,
        memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001,
        double_q=True,
        sess=sess
    )

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    observation = env.reset()
    # total_reward = 0

    while True:
        if total_steps - MEMORY_SIZE > 8000:
            env.render()
        action = RL.choose_action(observation)

        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10
        RL.store_transition(observation, action, reward, observation_)
        # total_reward += reward

        if total_steps > MEMORY_SIZE:
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:
            break

        # if total_steps % 300 == 0:
        #     print('reward: ', round(total_reward, 2))

        observation = observation_
        total_steps += 1


if __name__ == '__main__':
    train(natural_DQN)
    # train(double_DQN)
