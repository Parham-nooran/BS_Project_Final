import tensorflow as tf

from deeprl import explorations, logger, replays
from deeprl.tensorflow import agents, models, normalizers, updaters


def default_model(layers):
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(layers, 'relu'),
            head=models.DeterministicPolicyHead()),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP(layers, 'relu'),
            head=models.ValueHead()),
        observation_normalizer=normalizers.MeanStd())


class DDPG(agents.Agent):

    def __init__(self, model=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None, layers=(256, 256)):
        
        self.model = model or default_model(layers)
        self.replay = replay or replays.Buffer()
        self.exploration = exploration or explorations.NormalActionNoise()
        self.actor_updater = actor_updater or \
            updaters.DeterministicPolicyGradient()
        self.critic_updater = critic_updater or \
            updaters.DeterministicQLearning()

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(seed=seed)
        self.model.initialize(observation_space, action_space)
        self.replay.initialize(seed)
        self.exploration.initialize(self._policy, action_space, seed)
        self.actor_updater.initialize(self.model)
        self.critic_updater.initialize(self.model)

    def step(self, observations, steps):
        actions = self.exploration(observations, steps)

        self.last_observations = observations.copy()
        self.last_actions = actions.copy()

        return actions

    def test_step(self, observations, steps):
        return self._greedy_actions(observations).numpy()

    def update(self, observations, rewards, resets, terminations, steps):
        self.replay.store(
            observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards, resets=resets,
            terminations=terminations)

        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        if self.replay.ready(steps):
            self._update(steps)

        self.exploration.update(resets)

    @tf.function
    def _greedy_actions(self, observations):
        return self.model.actor(observations)

    def _policy(self, observations):
        return self._greedy_actions(observations).numpy()

    def _update(self, steps):
        keys = ('observations', 'actions', 'next_observations', 'rewards',
                'discounts')

        for batch in self.replay.get(*keys, steps=steps):
            infos = self._update_actor_critic(**batch)

            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + '/' + k, v.numpy())

        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

    @tf.function
    def _update_actor_critic(
        self, observations, actions, next_observations, rewards, discounts
    ):
        critic_infos = self.critic_updater(
            observations, actions, next_observations, rewards, discounts)
        actor_infos = self.actor_updater(observations)
        self.model.update_targets()
        return dict(critic=critic_infos, actor=actor_infos)
