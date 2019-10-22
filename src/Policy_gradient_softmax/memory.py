class EpisodeMemory:
    def __init__(self):
        self.ep_obs = []
        self.ep_as = []
        self.ep_rs = []

    def store_transactions(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def sample(self):
        return self.ep_obs, self.ep_as, self.ep_rs

    def sample_ep_obs(self):
        return self.ep_obs

    def sample_ep_as(self):
        return self.ep_as

    def sample_ep_rs(self):
        return self.ep_rs

    def clear(self):
        self.ep_rs.clear()
        self.ep_as.clear()
        self.ep_obs.clear()
