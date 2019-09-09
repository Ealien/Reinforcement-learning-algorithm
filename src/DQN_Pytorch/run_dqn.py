import os

import gym

from DQN_Pytorch.dqn import DQNAgent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = gym.make('CartPole-v1')

agent = DQNAgent(n_actions=env.action_space.n,
                 n_features=env.observation_space.shape[0],
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=100,
                 memory_size=2000,
                 batch_size=32,
                 e_greedy_increment=0.001)

total_step = 0

for i_episode in range(100):
    observation = env.reset()
    ep_r = 0
    while True:
        env.render()
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2

        agent.memory.store_transition(observation, action, reward, observation_)
        ep_r += reward
        if total_step > 500:
            agent.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(agent.epsilon, 2))
            break

        observation = observation_
        total_step += 1
env.close()
agent.plot_cost()
