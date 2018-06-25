import random
import numpy as np

class Noise(object):
    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        self.state = self.state + self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        return(self.state)
