import os
import gym.wrappers
import numpy as np
from deeprl import environments
from deeprl.utils import logger

def gym_environment(*args, **kwargs):
    
    def _builder(*args, **kwargs):
        return gym.make(*args, **kwargs)

    return build_environment(_builder, *args, **kwargs)


def bullet_environment(*args, **kwargs):

    def _builder(*args, **kwargs):
        import pybullet_envs  
        import parham_envs
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


def control_suite_environment(*args, **kwargs):

    def _builder(name, *args, **kwargs):
        domain, task = name.split('-')
        environment = ControlSuiteEnvironment(
            domain_name=domain, task_name=task, *args, **kwargs)
        time_limit = int(environment.environment._step_limit)
        return gym.wrappers.TimeLimit(environment, time_limit)

    return build_environment(_builder, *args, **kwargs)


def _flatten_observation(observation):
    observation = [np.array([o]) if np.isscalar(o) else o.ravel()
                   for o in observation.values()]
    return np.concatenate(observation, axis=0)


class ControlSuiteEnvironment(gym.core.Env):

    def __init__(
        self, domain_name, task_name, task_kwargs=None, visualize_reward=True,
        environment_kwargs=None
    ):
        from dm_control import suite
        self.environment = suite.load(domain_name=domain_name, task_name=task_name,
            task_kwargs=task_kwargs, visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs)

        observation_spec = self.environment.observation_spec()
        dim = sum([np.int(np.prod(spec.shape)) for spec in observation_spec.values()])
        high = np.full(dim, np.inf, np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        action_spec = self.environment.action_spec()
        self.action_space = gym.spaces.Box(action_spec.minimum, action_spec.maximum, dtype=np.float32)

    def seed(self, seed):
        self.environment.task._random = np.random.RandomState(seed)

    def step(self, action):
        try:
            time_step = self.environment.step(action)
            observation = _flatten_observation(time_step.observation)
            reward = time_step.reward

            done = time_step.last()
            if done:
                done = self.environment.task.get_termination(self.environment.physics)
                done = done is not None

            self.last_time_step = time_step

        except Exception as e:
            path = logger.get_path()
            os.makedirs(path, exist_ok=True)
            save_path = os.path.join(path, 'crashes.txt')
            error = str(e)
            with open(save_path, 'a') as file:
                file.write(error + '\n')
            logger.error(error)
            observation = _flatten_observation(self.last_time_step.observation)
            observation = np.zeros_like(observation)
            reward = 0.
            done = True

        return observation, reward, done, {}

    def reset(self):
        time_step = self.environment.reset()
        self.last_time_step = time_step
        return _flatten_observation(time_step.observation)

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array'
        return self.environment.physics.render(height=height, width=width, camera_id=camera_id)


Gym = gym_environment
Bullet = bullet_environment
ControlSuite = control_suite_environment