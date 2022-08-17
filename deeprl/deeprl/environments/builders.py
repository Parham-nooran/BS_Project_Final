import gym.wrappers

from deeprl import environments


def gym_environment(*args, **kwargs):
    
    def _builder(*args, **kwargs):
        return gym.make(*args, **kwargs)

    return build_environment(_builder, *args, **kwargs)


def bullet_environment(*args, **kwargs):

    def _builder(*args, **kwargs):
        import pybullet_envs  
        import ant_bullet_env
        return gym.make(*args, **kwargs)

    return build_environment(_builder, *args, **kwargs)



def build_environment(builder, name, terminal_timeouts=False, time_feature=False,
    max_episode_steps='default', scaled_actions=True, *args, **kwargs):
    

    environment = builder(name, *args, **kwargs)

    if max_episode_steps == 'default':
        max_episode_steps = environment._max_episode_steps

    if not terminal_timeouts:
        assert type(environment) == gym.wrappers.TimeLimit, environment
        environment = environment.env

    if time_feature:
        environment = environments.wrappers.TimeFeature(
            environment, max_episode_steps)

    if scaled_actions:
        environment = environments.wrappers.ActionRescaler(environment)

    environment.name = name
    environment.max_episode_steps = max_episode_steps

    return environment



Gym = gym_environment
Bullet = bullet_environment
