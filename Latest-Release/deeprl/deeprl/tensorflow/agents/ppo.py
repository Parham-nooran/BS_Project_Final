import tensorflow as tf

from deeprl import logger
from deeprl.tensorflow import agents, updaters


class PPO(agents.A2C):

    def __init__(self, model=None, replay=None, actor_updater=None, critic_updater=None, layers=(64, 64)):

        actor_updater = actor_updater or updaters.ClippedRatio()
        super().__init__(
            model=model, replay=replay, actor_updater=actor_updater,
            critic_updater=critic_updater, layers=layers)

    def _update(self):
        batch = self.replay.get_full('observations', 'next_observations')
        values, next_values = self._evaluate(**batch)
        values, next_values = values.numpy(), next_values.numpy()
        self.replay.compute_returns(values, next_values)

        train_actor = True
        actor_iterations = 0
        critic_iterations = 0
        keys = 'observations', 'actions', 'advantages', 'log_probs', 'returns'

        for batch in self.replay.get(*keys):
            if train_actor:
                infos = self._update_actor_critic(**batch)
                actor_iterations += 1
            else:
                batch = {k: batch[k] for k in ('observations', 'returns')}
                infos = dict(critic=self.critic_updater(**batch))
            critic_iterations += 1

            if train_actor:
                train_actor = not infos['actor']['stop'].numpy()

            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + '/' + k, v.numpy())

        logger.store('actor/iterations', actor_iterations)
        logger.store('critic/iterations', critic_iterations)

        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

    @tf.function
    def _update_actor_critic(
        self, observations, actions, advantages, log_probs, returns
    ):
        actor_infos = self.actor_updater(
            observations, actions, advantages, log_probs)
        critic_infos = self.critic_updater(observations, returns)
        return dict(actor=actor_infos, critic=critic_infos)
