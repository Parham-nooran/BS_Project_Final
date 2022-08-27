from gym.envs.registration import register


register(
    id='RexBulletEnv-v0',
    entry_point='rex_gym.envs:RexWalkEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0
)
