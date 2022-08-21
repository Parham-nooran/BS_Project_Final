import tensorflow as tf

from deeprl import logger, replays
from deeprl.tensorflow import agents, models, normalizers, updaters


def default_model(layers):
    return models.ActorCritic(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(layers, 'tanh'),
            head=models.DetachedScaleGaussianPolicyHead()),
        critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(layers, 'tanh'),
            head=models.ValueHead()),
        observation_normalizer=normalizers.MeanStd())


class A2C(agents.Agent):

    def __init__(self, model=None, replay=None, actor_updater=None, critic_updater=None, layers=(64, 64)):
        self.model = model or default_model(layers)
        self.replay = replay or replays.Segment()
        self.actor_updater = actor_updater or updaters.StochasticPolicyGradient()
        self.critic_updater = critic_updater or updaters.VRegression()

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(seed=seed)
        self.model.initialize(observation_space, action_space)
        self.replay.initialize(seed)
        self.actor_updater.initialize(self.model)
        self.critic_updater.initialize(self.model)

    def step(self, observations, steps):
        actions, log_probs = self._step(observations)
        actions = actions.numpy()
        log_probs = log_probs.numpy()

        self.last_observations = observations.copy()
        self.last_actions = actions.copy()
        self.last_log_probs = log_probs.copy()

        return actions

    def test_step(self, observations, steps):
        return self._test_step(observations).numpy()

    def update(self, observations, rewards, resets, terminations, steps):
        self.replay.store(
            observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards, resets=resets,
            terminations=terminations, log_probs=self.last_log_probs)

        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        if self.replay.ready():
            self._update()

    @tf.function
    def _step(self, observations):
        distributions = self.model.actor(observations)
        if hasattr(distributions, 'sample_with_log_prob'):
            actions, log_probs = distributions.sample_with_log_prob()
        else:
            actions = distributions.sample()
            log_probs = distributions.log_prob(actions)
        return actions, log_probs

    @tf.function
    def _test_step(self, observations):
        return self.model.actor(observations).sample()

    @tf.function
    def _evaluate(self, observations, next_observations):
        values = self.model.critic(observations)
        next_values = self.model.critic(next_observations)
        return values, next_values

    def _update(self):
        batch = self.replay.get_full('observations', 'next_observations')
        values, next_values = self._evaluate(**batch)
        values, next_values = values.numpy(), next_values.numpy()
        self.replay.compute_returns(values, next_values)

        keys = 'observations', 'actions', 'advantages', 'log_probs'
        batch = self.replay.get_full(*keys)
        infos = self.actor_updater(**batch)
        for k, v in infos.items():
            logger.store('actor/' + k, v.numpy())

        for batch in self.replay.get('observations', 'returns'):
            infos = self.critic_updater(**batch)
            for k, v in infos.items():
                logger.store('critic/' + k, v.numpy())

        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()
