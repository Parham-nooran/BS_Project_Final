from deeprl import replays
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
            head=models.DistributionalValueHead(-150., 150., 51)),
        observation_normalizer=normalizers.MeanStd())


class TD4(agents.TD3):
    def __init__(self, model=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None, delay_steps=2, layers=(256, 256, 256)):

        model = model or default_model(layers)
        replay = replay or replays.Buffer(return_steps=5)
        actor_updater = actor_updater or \
            updaters.DistributionalDeterministicPolicyGradient()
        critic_updater = critic_updater or \
            updaters.TwinCriticDistributionalDeterministicQLearning()
        super().__init__(
            model=model, replay=replay, exploration=exploration,
            actor_updater=actor_updater, critic_updater=critic_updater,
            delay_steps=delay_steps, layer1=layer1, layer2=layer2, layer3=layer3)
