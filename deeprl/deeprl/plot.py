import argparse
import itertools
import os
import pathlib
import time

import matplotlib as mpl
from matplotlib import gridspec
from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deeprl import logger


def smooth(vals, window):

    if window > 1:
        if window > len(vals):
            window = len(vals)
        y = np.ones(window)
        x = vals
        z = np.ones(len(vals))
        mode = 'same'
        vals = np.convolve(x, y, mode) / np.convolve(z, y, mode)

    return vals


def stats(xs, means, stds):
    lengths = [len(x) for x in xs]
    min_length = min(lengths)
    xs = [x[:min_length] for x in xs]
    assert len(xs) == 1 or np.all(xs[1:] == xs[0]), xs
    x = xs[0]
    means = np.array([mean[:min_length] for mean in means])
    mean = means.mean(axis=0)
    min_mean = means.min(axis=0)
    max_mean = means.max(axis=0)
    if stds is not None:
        stds = np.array([std[:min_length] for std in stds])
        variances = stds ** 2
        var_within = variances.mean(axis=0)
        var_between = ((means - mean) ** 2).mean(axis=0)
        std = np.sqrt(var_within + var_between)
    else:
        std = None
    return x, mean, min_mean, max_mean, std


def flip(items, columns):
    return itertools.chain(*[items[i::columns] for i in range(columns)])


def get_data(paths, x_axis, y_axis, x_min, x_max, window):
    data = {}
    log_paths = []
    for path in paths:
        if os.path.isdir(path):
            log_paths.extend(pathlib.Path(path).rglob('log.*'))
        elif path[-7:-3] == 'log.':
            log_paths.append(path)

    for path in log_paths:
        sub_path, file = os.path.split(path)
        dfs = {}
        env, agent, seed = sub_path.split(os.sep)[-3:]
        df_seed = pd.read_csv(path, sep=',', engine='python')
        x = df_seed[x_axis].values
        if not np.all(np.diff(x) > 0):
            logger.warning(f'Skipping unsorted {env} {agent} {seed}')
            continue
        if x_min and x[-1] < x_min:
            logger.warning(
                f'Skipping {env} {agent} {seed} ({x[-1]} steps)')
            continue
        dfs[seed] = df_seed
        for seed, df in dfs.items():
            if env not in data:
                data[env] = {}
            if agent not in data[env]:
                data[env][agent] = {}
            assert seed not in data[env][agent]
            x = df[x_axis].values
            if y_axis in df:
                mean = df[y_axis].values
                if y_axis[-5:] == '/mean' and y_axis[:-5] + '/std' in df:
                    std = df[y_axis[:-5] + '/std'].values
                else:
                    std = None
            elif y_axis + '/mean' in df:
                mean = df[y_axis + '/mean'].values
                if y_axis + '/std' in df:
                    std = df[y_axis + '/std'].values
                else:
                    std = None
            else:
                raise KeyError(f'Key {y_axis} not found.')
            if x_max:
                max_index = np.argmax(x > x_max) or len(x)
            else:
                max_index = len(x)
            x = x[:max_index]
            mean = smooth(mean, window)
            mean = mean[:max_index]
            if std is not None:
                std = smooth(std, window)
                std = std[:max_index]
            data[env][agent][seed] = x, mean, std
    for env, env_data in data.items():
        for agent, agent_data in env_data.items():
            xs, means, stds = [], [], []
            for seed, (x, mean, std) in agent_data.items():
                xs.append(x)
                means.append(mean)
                stds.append(std)
            if stds[0] is None:
                stds = None
            print(env, agent)
            env_data[agent] = dict(
                seeds=(xs, means), stats=stats(xs, means, stds))
    return data


def plot(paths, fig, x_axis='train/steps', y_axis='test/episode_score', x_label=None, y_label=None, window=1, interval='bounds', show_seeds=False,
    columns=None, x_min=None, x_max=None, y_min=None, y_max=None, name=None,
    cmap=None, legend_columns=None, legend_marker_size=None, dpi=150, title=None):
    logger.log('Loading data...')
    data = get_data(
        paths, x_axis, y_axis, x_min, x_max,
        window)
    envs = sorted(data.keys(), key=str.casefold)
    num_envs = len(envs)
    if num_envs == 0:
        logger.error('No logs found.')
        return
    agents = set()
    for env in data:
        for agent in data[env]:
            agents.add(agent)
    agents = sorted(agents, key=str.casefold)
    num_agents = len(agents)
    if not cmap:
        if num_agents <= 10:
            cmap = 'tab10'
        elif num_agents <= 20:
            cmap = 'tab20'
        else:
            cmap = 'rainbow'
    cmap = plt.get_cmap(cmap)
    if isinstance(cmap, mpl.colors.ListedColormap):
        colors = cmap(range(num_agents))
    else:
        colors = list(cmap(np.linspace(0, 1, num_agents)))
    agent_colors = {a: c for a, c in zip(agents, colors)}
    if columns is None:
        columns = int(np.ceil(np.sqrt(num_envs)))
    else:
        columns = min(columns, num_envs)
    rows = int(np.ceil(num_envs / columns))
    if fig is None:
        plt.ion()
        fig = plt.figure(figsize=(columns * 6, rows * 5))
    else:
        fig.clear()
    grid = gridspec.GridSpec(
        rows + 1, 1 + columns, height_ratios=[1] * rows + [0.1],
        width_ratios=[0] + [1] * columns)
    axes = []
    for i in range(num_envs):
        ax = fig.add_subplot(grid[i // columns, 1 + i % columns])
        axes.append(ax)
    logger.log('Plotting...')
    for env, ax in zip(envs, axes):
        if interval in ['std', 'bounds']:
            for agent in sorted(data[env], key=str.casefold):
                color = agent_colors[agent]
                x, mean, min_mean, max_mean, std = data[env][agent]['stats']
                if interval == 'std':
                    if std is None:
                        logger.error('No std found in the data.')
                    else:
                        ax.fill_between(
                            x, mean - std, mean + std, color=color, alpha=0.1,
                            lw=0)
                elif interval == 'bounds':
                    ax.fill_between(
                        x, min_mean, max_mean, color=color, alpha=0.1, lw=0)
        for agent in sorted(data[env], key=str.casefold):
            color = agent_colors[agent]
            xs, means = data[env][agent]['seeds']
            if show_seeds and len(xs) > 1:
                for x, mean in zip(xs, means):
                    ax.plot(x, mean, c=color, lw=1, alpha=0.5)
            x, mean = data[env][agent]['stats'][:2]
            ax.plot(x, mean, c=color, lw=2, alpha=1)
        ax.set_ylim(ymin=y_min, ymax=y_max)
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', tight=True, nbins=6)
        ax.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: f'{x:,g}'))
        low, high = ax.get_xlim()
        if max(abs(low), abs(high)) >= 1e3:
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.grid(linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        if x_label is None:
            x_label = 'Steps' if x_axis == 'train/steps' else x_axis
        if y_label is None:
            y_label = 'Score' if y_axis == 'test/episode_score' else y_axis
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        else:
            ax.set_title(env)
    legend_ax = fig.add_subplot(grid[-1:, :])
    legend_ax.set_axis_off()
    handles = []
    for color in colors:
        marker = lines.Line2D(range(1), range(1), marker='o', markerfacecolor=color,
            markersize=legend_marker_size, linewidth=0, markeredgewidth=0)
        handles.append(marker)

    if legend_columns is None:
        legend_columns = range(num_agents, 0, -1)
    else:
        legend_columns = [legend_columns]
    for ncol in legend_columns:
        legend = legend_ax.legend(
            flip(handles, ncol), flip(agents, ncol), loc='center',
            mode='expand', borderaxespad=0, borderpad=0, handlelength=0.9,
            ncol=ncol, numpoints=1)
        legend_frame = legend.get_frame()
        legend_frame.set_linewidth(0)
        fig.tight_layout(pad=0, w_pad=0, h_pad=1.0)
        fig.canvas.draw()
        renderer = legend_ax.get_renderer_cache()
        h_packer = legend.get_children()[0].get_children()[1]
        target_width = h_packer.get_extent(renderer)[0]
        current_width = sum(
            [ch.get_extent(renderer)[0] for ch in h_packer.get_children()])
        if target_width > 1.3 * current_width:
            break
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+', default=[])
    args = parser.parse_args()
    has_gui = True
    plt.rc('font', family='serif', size=12)
    seconds = 0
    start_time = time.time()
    fig = plot(**vars(args), fig=None)
    try:
        if seconds == 0:
            if has_gui:
                while plt.get_fignums() != []:
                    plt.pause(0.1)
        else:
            while True:
                if has_gui:
                    while time.time() - start_time < seconds:
                        plt.pause(0.1)
                        assert plt.get_fignums() != []
                else:
                    time.sleep(seconds)
                start_time = time.time()
                plot(**vars(args), fig=fig)
    except Exception:
        pass