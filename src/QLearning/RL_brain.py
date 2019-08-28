import numpy as np
import pandas as pd


class QLearningTable:
    # 初始化
    def __init__(self, actions, reward_decay=0.9, learning_rate=0.01, e_greedy=0.9):
        # 所有动作
        self.actions = actions
        # 学习率
        self.lr = learning_rate
        # 折扣因子
        self.gamma = reward_decay
        # 贪婪度
        self.epsilon = e_greedy
        # Q表
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # 选择动作
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # 动作选择
        if np.random.uniform() < self.epsilon:
            state_actions = self.q_table.loc[observation, :]
            action = np.random.choice(
                state_actions[state_actions == np.max(state_actions)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    # 学习
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新

    # 检测状态是否存在Q表中
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 加入新状态到Q表
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
