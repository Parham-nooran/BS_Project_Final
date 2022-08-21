from gym.envs.registration import register

register(
    id='AntBulletEnv-v5', 
    entry_point='parham_envs.envs:AntBulletEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0
)

register(
    id='LaikagoBulletEnv-v0',
    entry_point='parham_envs.envs:LaikagoBulletEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0
)

register(
    id='QuadrupedBulletEnv-v0',
    entry_point='parham_envs.envs:QuadrupedBulletEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0
)