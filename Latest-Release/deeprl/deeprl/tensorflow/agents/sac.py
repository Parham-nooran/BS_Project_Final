import tensorflow as tf

from deeprl import explorations
from deeprl.tensorflow import agents, models, normalizers, updaters


def default_model(layers):
    return models.ActorTwinCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(layers, 'relu'),
            head=models.GaussianPolicyHead(loc_activation=None,
                distribution=models.SquashedMultivariateNormalDiag)),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP(layers, 'relu'),
            head=models.ValueHead()),
        observation_normalizer=normalizers.MeanStd())


class SAC(agents.DDPG):

    def __init__(self, model=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None, layers=(256, 256)):

        model = model or default_model(layers=layers)
        exploration = exploration or explorations.NoActionNoise()
        actor_updater = actor_updater or \
            updaters.TwinCriticSoftDeterministicPolicyGradient()
        critic_updater = critic_updater or updaters.TwinCriticSoftQLearning()
        super().__init__(
            model=model, replay=replay, exploration=exploration,
            actor_updater=actor_updater, critic_updater=critic_updater, layers=layers)

    @tf.function
    def _stochastic_actions(self, observations):
        return self.model.actor(observations).sample()

    def _policy(self, observations):
        return self._stochastic_actions(observations).numpy()

    @tf.function
    def _greedy_actions(self, observations):
        return self.model.actor(observations).mode()
