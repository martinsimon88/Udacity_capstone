from random import *
import numpy as np


class ReplayMemory:
    def __init__(self, capacity, resolution):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

    def get_sequence(self, sequence_length, sample_size):

        # Random start indexes
        i = sample(range(0, self.size - sequence_length), sample_size)

        s1 = []
        a = []
        s2 = []
        isterminal = []
        r = []

        for start in i:
            end = min(start + sequence_length, self.size)
            s1.append([self.s1[start:end]])
            a.append([self.a[start:end]])
            s2.append([self.s2[start:end]])
            isterminal.append([self.isterminal[start:end]])
            r.append([self.r[start:end]])

        return s1, a, s2, isterminal, r


class Episode:
    def __init__(self):
        self.s1 = []
        self.s2 = []
        self.a = []
        self.r = []
        self.isterminal = []
        self.length = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1.append(s1)
        self.s2.append(s2)
        self.a.append(action)
        self.r.append(reward)
        self.isterminal.append(isterminal)
        self.length += 1


class SequentialReplayMemory:
    def __init__(self, capacity, sequence_length):
        self.episodes = []
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.sequence_length = sequence_length

    def add_episodes(self, episode):
        if episode.length < self.sequence_length:
            return
        if len(self.episodes) >= self.capacity:
            self.pos = (self.pos + 1) % self.capacity
            self.episodes[self.pos] = episode
        else:
            self.episodes.append(episode)
        self.size = min(self.size + 1, self.capacity)

    def get_random_episode(self):
        return choice(self.episodes)

    def get_sequence(self):
        e = choice(self.episodes)
        start = int(random() * (e.length - self.sequence_length))
        end = min(start + self.sequence_length, e.length)
        return e.s1[start:end], e.a[start:end], e.s2[start:end], e.isterminal[start:end], e.r[start:end]

