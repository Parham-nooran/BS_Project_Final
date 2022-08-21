import argparse
import os

import deeprl
import yaml


def train(header, agent, environment, test_environment, trainer, before_training, after_training,
            parallel, sequential, seed, name, environment_name, checkpoint, path):

    args = dict(locals())

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
        environment = environment or config.test_environment
        environment = environment or config.environment
        trainer = trainer or config.trainer

    if header:
        exec(header)

    _environment = environment
    environment = deeprl.environments.distribute(lambda: eval(_environment), parallel, sequential)
    environment.initialize(seed=seed)

    _test_environment = test_environment if test_environment else _environment
    test_environment = deeprl.environments.distribute(lambda: eval(_test_environment))
    test_environment.initialize(seed=seed + 10000)

    if not agent:
        raise ValueError('No agent specified.')
    
    agent = eval(agent)
    agent.initialize(observation_space=environment.observation_space,
        action_space=environment.action_space, seed=seed)

    if checkpoint_path:
        agent.load(checkpoint_path)

    if not environment_name:
        if hasattr(test_environment, 'name'):
            environment_name = test_environment.name
        else:
            environment_name = test_environment.__class__.__name__
    if not name:
        if hasattr(agent, 'name'):
            name = agent.name
        else:
            name = agent.__class__.__name__
        if parallel != 1 or sequential != 1:
            name += f'-{parallel}x{sequential}'
    path = os.path.join(environment_name, name, str(seed))
    deeprl.logger.initialize(path, script_path=__file__, config=args)

    trainer = trainer or 'deeprl.Trainer()'
    trainer = eval(trainer)
    trainer.initialize(agent=agent, environment=environment,
        test_environment=test_environment)

    if before_training:
        exec(before_training)

    trainer.run()

    if after_training:
        exec(after_training)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--header')
    parser.add_argument('--agent')
    parser.add_argument('--environment', '--env')
    parser.add_argument('--test_environment', '--test_env')
    parser.add_argument('--trainer')
    parser.add_argument('--before_training')
    parser.add_argument('--after_training')
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--sequential', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name')
    parser.add_argument('--environment_name')
    parser.add_argument('--checkpoint', default='last')
    parser.add_argument('--path')

    args = vars(parser.parse_args())
    train(**args)
