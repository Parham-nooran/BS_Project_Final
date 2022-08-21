import setuptools


setuptools.setup(
    name='deeprl',
    description='Deep Reinforcement Learning Algorithms Implementation',
    url='https://github.com/',
    version='0.0.1',
    author='Parham Nooranbakht',
    author_email='parhamnooranbakht@email.kntu.ac.ir',
    install_requires=[
        'gym', 'matplotlib', 'numpy', 'pybullet', 'pandas', 'pyyaml', 'termcolor'],
    license='MIT',
    python_requires='>=3.6',
    keywords=['deep reinforcement learning', 'reinforcement learning', 'deep learning'])
