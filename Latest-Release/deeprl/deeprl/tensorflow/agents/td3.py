from deeprl import logger
from deeprl.tensorflow import agents, models, normalizers, updaters


def default_model(layers):
    return models.ActorTwinCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(layers, 'relu'),
            head=models.DeterministicPolicyHead()),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP(layers, 'relu'),
            head=models.ValueHead()),
        observation_normalizer=normalizers.MeanStd())


class TD3(agents.DDPG):

    def __init__(self, model=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None, delay_steps=2, layers=(256, 256)):

        model = model or default_model(layers=layers)
        critic_updater = critic_updater or \
            updaters.TwinCriticDeterministicQLearning()
        super().__init__(
            model=model, replay=replay, exploration=exploration,
            actor_updater=actor_updater, critic_updater=critic_updater, layers=layers)
        self.delay_steps = delay_steps
        self.model.critic = self.model.critic_1

    def _update(self, steps):
        keys = ('observations', 'actions', 'next_observations', 'rewards',
                'discounts')
        for i, batch in enumerate(self.replay.get(*keys, steps=steps)):
            if (i + 1) % self.delay_steps == 0:
                infos = self._update_actor_critic(**batch)
            else:
                infos = dict(critic=self.critic_updater(**batch))
            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + '/' + k, v.numpy())

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()
