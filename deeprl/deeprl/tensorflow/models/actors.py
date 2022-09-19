import tensorflow as tf
import tensorflow_probability as tfp

from deeprl.tensorflow import models


FLOAT_EPSILON = 1e-8

# A2C, TRPO, PPO
class DetachedScaleGaussianPolicyHead(tf.keras.Model): 
    def __init__(self, loc_activation='tanh', dense_loc_kwargs=None, log_scale_init=0.,
        scale_min=1e-4, scale_max=1., distribution=tfp.distributions.MultivariateNormalDiag):

        super().__init__()
        self.loc_activation = loc_activation
        if dense_loc_kwargs is None:
            dense_loc_kwargs = models.default_dense_kwargs()
        self.dense_loc_kwargs = dense_loc_kwargs
        self.log_scale_init = log_scale_init
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.distribution = distribution

    def initialize(self, action_size):
        self.loc_layer = tf.keras.layers.Dense(
            action_size, self.loc_activation, **self.dense_loc_kwargs)
        log_scale = [[self.log_scale_init] * action_size]
        self.log_scale = tf.Variable(log_scale, dtype=tf.float32)

    def call(self, inputs):
        loc = self.loc_layer(inputs)
        batch_size = tf.shape(inputs)[0]
        scale = tf.math.softplus(self.log_scale) + FLOAT_EPSILON
        scale = tf.clip_by_value(scale, self.scale_min, self.scale_max)
        scale = tf.tile(scale, (batch_size, 1))
        return self.distribution(loc, scale)

# MPO
class GaussianPolicyHead(tf.keras.Model):
    def __init__(
        self, loc_activation='tanh', dense_loc_kwargs=None,
        scale_activation='softplus', scale_min=1e-4, scale_max=1,
        dense_scale_kwargs=None,
        distribution=tfp.distributions.MultivariateNormalDiag
    ):
        super().__init__()
        self.loc_activation = loc_activation
        if dense_loc_kwargs is None: dense_loc_kwargs = models.default_dense_kwargs()
        self.dense_loc_kwargs = dense_loc_kwargs
        self.scale_activation = scale_activation
        self.scale_min = scale_min
        self.scale_max = scale_max
        if dense_scale_kwargs is None: dense_scale_kwargs = models.default_dense_kwargs()
        self.dense_scale_kwargs = dense_scale_kwargs
        self.distribution = distribution

    def initialize(self, action_size):
        self.loc_layer = tf.keras.layers.Dense(action_size, self.loc_activation, **self.dense_loc_kwargs)
        self.scale_layer = tf.keras.layers.Dense(action_size, self.scale_activation, **self.dense_scale_kwargs)

    def call(self, inputs):
        loc = self.loc_layer(inputs)
        scale = self.scale_layer(inputs)
        scale = tf.clip_by_value(scale, self.scale_min, self.scale_max)
        return self.distribution(loc, scale)

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
