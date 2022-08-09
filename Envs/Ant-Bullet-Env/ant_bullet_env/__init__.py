from gym.envs.registration import register

register(
    id='AntBulletEnv-v5', 
    entry_point='ant_bullet_env.envs:AntBulletEnv',
    max_episode_steps=1500,
    reward_threshold=15000.0
)
