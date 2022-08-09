import tensorflow as tf
import numpy as np
import random

class Actor(tf.keras.Model):
    def __init__(self, encoder, torso, head):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(
        self, observation_space, action_space, observation_normalizer=None
        ):
        self.encoder.initialize(observation_normalizer)
        self.head.initialize(action_space.shape[0])
    
    def call(self, *inputs):
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)

class Agent:
    def initialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
    
    

class A2C:
    pass