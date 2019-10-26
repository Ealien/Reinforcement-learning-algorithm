import numpy as np


class Memory:
    def __init__(self, s_dim, a_dim, memory_size=5000, batch_size=32):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = np.zeros((memory_size, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.memory_count = 0

    def store_transitions(self, s, s_, r, a):
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_count % self.memory_size
        self.memory[index, :] = transition
        self.memory_count += 1

    def sample_batch(self):
        if self.memory_count > self.memory_size:
            indices = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            indices = np.random.choice(self.memory_count, size=self.batch_size)
        return self.memory[indices, :]
