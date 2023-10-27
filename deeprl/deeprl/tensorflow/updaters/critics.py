import tensorflow as tf

from deeprl.tensorflow import updaters

# A2C, PPO, TRPO
class VRegression:
    def __init__(self, loss=None, optimizer=None, gradient_clip=0):
        self.loss = loss or tf.keras.losses.MeanSquaredError()
        self.optimizer = optimizer or tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.critic.trainable_variables

    @tf.function
    def __call__(self, observations, returns):
        with tf.GradientTape() as tape:
            values = self.model.critic(observations)
            loss = self.loss(returns, values)

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(
                gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss, v=values)

# MPO
class ExpectedSARSA:
    def __init__(
        self, num_samples=20, loss=None, optimizer=None, gradient_clip=0
    ):
        self.num_samples = num_samples
        self.loss = loss or tf.keras.losses.MeanSquaredError()
        self.optimizer = optimizer or \
            tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-8)
        self.gradient_clip = gradient_clip

    def initialize(self, model):
        self.model = model
        self.variables = self.model.critic.trainable_variables

    @tf.function
    def __call__(self, observations, actions, next_observations, rewards, discounts):
        next_target_distributions = self.model.target_actor(next_observations)
        
        next_actions = next_target_distributions.sample(self.num_samples)
        next_actions = updaters.merge_first_two_dims(next_actions)
        
        next_observations = updaters.tile(next_observations, self.num_samples)
        next_observations = updaters.merge_first_two_dims(next_observations)

        next_values = self.model.target_critic(next_observations, next_actions)
        next_values = tf.reshape(next_values, (self.num_samples, -1))
        next_values = tf.reduce_mean(next_values, axis=0)
        
        returns = rewards + discounts * next_values

        with tf.GradientTape() as tape:
            values = self.model.critic(observations, actions)
            loss = self.loss(returns, values)

        gradients = tape.gradient(loss, self.variables)
        if self.gradient_clip > 0:
            gradients = tf.clip_by_global_norm(gradients, self.gradient_clip)[0]
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return dict(loss=loss, q=values)