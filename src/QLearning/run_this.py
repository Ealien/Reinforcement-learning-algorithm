from QLearning.RL_brain import QLearningTable
from QLearning.maze_env import Maze


def update():
    for episode in range(100):
        # 初始化状态
        observation = env.reset()

        while True:
            # 刷新环境
            env.render()

            # RL选择动作执行
            action = RL.choose_action(str(observation))

            observation_, reward, done = env.step(action)

            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_

            if done:
                break

    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
