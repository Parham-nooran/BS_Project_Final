from gym.envs.registration import register

register(
    id='AntBulletEnv-v5', 
    entry_point='ant_bullet_env.envs:AntBulletEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0
)
