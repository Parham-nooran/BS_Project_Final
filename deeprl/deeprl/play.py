import os
import yaml
import deeprl  
import argparse
from matplotlib import animation
import matplotlib.pyplot as plt

def play_gym(agent, environment, save_as_gif):
    frames = []
    environment = deeprl.environments.distribute(lambda: environment)

    observations = environment.start()
    if save_as_gif:
        frames.append(environment.render(mode="rgb_array")[0])
    else:
        environment.render()

    score = 0
    length = 0
    min_reward = float('inf')
    max_reward = -float('inf')
    global_min_reward = float('inf')
    global_max_reward = -float('inf')
    steps = 0
    episodes = 0
    
    # for i in range(100):
    while True:
        actions = agent.test_step(observations, steps)
        observations, infos = environment.step(actions)
        agent.test_update(**infos, steps=steps)
        if save_as_gif:
            frames.append(environment.render(mode="rgb_array")[0])
        else:
            environment.render()

        steps += 1
        reward = infos['rewards'][0]
        score += reward
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)
        global_min_reward = min(global_min_reward, reward)
        global_max_reward = max(global_max_reward, reward)
        length += 1
        if infos['resets'][0]:
            term = infos['terminations'][0]
            episodes += 1

            print()
            print(f'Episodes: {episodes:,}')
            print(f'Score: {score:,.3f}')
            print(f'Length: {length:,}')
            print(f'Terminal: {term:}')
            print(f'Min reward: {min_reward:,.3f}')
            print(f'Max reward: {max_reward:,.3f}')
            print(f'Global min reward: {min_reward:,.3f}')
            print(f'Global max reward: {max_reward:,.3f}')

            score = 0
            length = 0
            min_reward = float('inf')
            max_reward = -float('inf')
    
    return frames

def save_frames_as_gif(frames, path='deeprl/deeprl/', filename='gym_animation.gif'):
    print(frames[0].shape)
    plt.figure(figsize=(frames[0].shape[1]/20.0, frames[0].shape[0]/20.0), dpi=70)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=200)
    anim.save(path + filename, writer='imagemagick', fps=100)

def play(path, checkpoint, seed, header, agent, environment, save_as_gif=False):

    checkpoint_path = None
    
    if path:
        deeprl.logger.log(f'Loading experiment from {path}')

        if checkpoint == 'none' or agent is not None:
            deeprl.logger.log('Not loading any weights')

        else:
            checkpoint_path = os.path.join(path, 'checkpoints')
            if not os.path.isdir(checkpoint_path):
                deeprl.logger.error(f'{checkpoint_path} is not a directory')
                checkpoint_path = None

            checkpoint_ids = []
            for file in os.listdir(checkpoint_path):
                if file[:5] == 'step_':
                    checkpoint_id = file.split('.')[0]
                    checkpoint_ids.append(int(checkpoint_id[5:]))

            if checkpoint_ids:
                if checkpoint == 'last':
                    checkpoint_id = max(checkpoint_ids)
                    checkpoint_path = os.path.join(
                        checkpoint_path, f'step_{checkpoint_id}')

                else:
                    checkpoint_id = int(checkpoint)
                    if checkpoint_id in checkpoint_ids:
                        checkpoint_path = os.path.join(
                            checkpoint_path, f'step_{checkpoint_id}')
                    else:
                        deeprl.logger.error(f'Checkpoint {checkpoint_id} '
                                           f'not found in {checkpoint_path}')
                        checkpoint_path = None

            else:
                deeprl.logger.error(f'No checkpoint found in {checkpoint_path}')
                checkpoint_path = None

        arguments_path = os.path.join(path, 'config.yaml')
        with open(arguments_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = argparse.Namespace(**config)

        header = header or config.header
        agent = agent or config.agent
        environment = environment or config.environment

    if header:
        exec(header)

    if not agent:
        raise ValueError('No agent specified.')
    agent = eval(agent)

    environment = eval(environment)
    environment.seed(seed)

    agent.initialize(observation_space=environment.observation_space,
        action_space=environment.action_space, seed=seed)

    if checkpoint_path:
        agent.load(checkpoint_path)

    if isinstance(environment, deeprl.environments.wrappers.ActionRescaler):
        environment_type = environment.env.__class__.__name__
    else:
        environment_type = environment.__class__.__name__

    environment.render()

    frames = play_gym(agent, environment, save_as_gif=save_as_gif)
    if save_as_gif:
        save_frames_as_gif(frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--checkpoint', default='last')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--header')
    parser.add_argument('--agent')
    parser.add_argument('--environment', '--env')
    args = vars(parser.parse_args())
    play(**args)
