import numpy as np
import matplotlib.pyplot as plt


def normal_obs(mean, variance, n=1):
    '''Returns a single observation from a normal random variable with
    mean and variance provided.'''
    return np.random.normal(mean, np.sqrt(variance), n)


def figure_generation_one(coords, f1, f2, reset=None):
    '''
    Generates a figure for plotting the stochastic process.
    ---
    coords (tuple of numpy arrays): coords[0] and coords[1] must consist
    of coordinatesb to be plotted. This is used for setting the xlim and ylim.

    f1 (float): the length of the figure.
    f2 (float): the height of the figure.

    reset (None or float): the reset position of the stochastic process
    if a reset is present.
    '''
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(f1, f2))
    ax.set_xlim([0, coords[0][-1]])
    min_val = np.min(coords[1])
    max_val = np.max(coords[1])

    # Setting ylim
    if min_val >= 0:
        ax.set_ylim([-max_val, max_val])
    elif max_val <= 0:
        ax.set_ylim([min_val, -min_val])
    else:
        val = np.max([-min_val, max_val])
        ax.set_ylim([-val, val])
    reset_val = coords[1][0]

    # Reset Line
    if reset is not None:
        reset_val = reset
    ax.hlines(reset_val, xmin=0, xmax=coords[0][-1],
              linestyle='dashed',
              color='green')
    return fig, ax
