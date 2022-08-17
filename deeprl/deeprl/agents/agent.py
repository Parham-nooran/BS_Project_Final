import abc


class Agent(abc.ABC):
    def initialize(self, observation_space, action_space, seed=None):
        pass

    @abc.abstractmethod
    def step(self, observations, steps):
        pass

    def update(self, observations, rewards, resets, terminations, steps):
        pass

    @abc.abstractmethod
    def test_step(self, observations, steps):
        pass

    def test_update(self, observations, rewards, resets, terminations, steps):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
