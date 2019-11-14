import numpy as np
import random
from collections import namedtuple, deque

class OUNoise:
    def __init__(self, size, phi, theta, psi):
        self.phi = phi * np.ones(size)
        self.theta = theta
        self.psi = psi
        self.reset()

    def reset(self):
        self.state = self.phi

    def sample(self):
        x = self.state
        dx = self.theta * (self.phi - x) + self.psi * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class Buffer():
    def __init__(self, buffer_size, batch):
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch = batch #size of batch
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch=64):
        return random.sample(self.memory, k=self.batch)

    def __len__(self):
        return len(self.memory)