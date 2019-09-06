from Deep_Q_Network.RL_brain import DeepQNetwork
# from Deep_Q_Network.DQN_modified import DeepQNetwork
from Deep_Q_Network.maze_env import Maze


def run_maze():
    step = 0
    for episode in range(300):
        # 初始化状态
        observation = env.reset()

        while True:
            env.render()
            # 根据状态选择动作
            action = RL.choose_action(observation)
            # 在环境中执行动作，获取下一个状态和奖励值
            observation_, reward, done = env.step(action)
            # 存储状态、动作和奖励值
            RL.store_transition(observation, action, reward, observation_)

            # 在记忆中的信息积累足够多时进行训练
            if step > 200 and step % 5 == 0:
                RL.learn()

            observation = observation_

            if done:
                break
            step += 1
    print("game over")
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.98,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.005,
                      output_graph=True
                      )

    env.after(100, run_maze)
    env.mainloop()
    # 神经网络的误差曲线
    RL.plot_cost()
