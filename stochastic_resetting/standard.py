import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


def normal_obs(mean, variance):
    return np.random.normal(mean, np.sqrt(variance))


def poisson_obs(r):
    return np.random.poisson(r)


def figure_generation_one(coords, f1, f2, reset=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize = (f1, f2))
    min_val = np.min(coords[1])
    max_val = np.max(coords[1])
    if min_val >= 0:
        ax.set_ylim([-max_val, max_val])
    elif max_val <= 0:
        ax.set_ylim([min_val, -min_val])
    else:
        val = np.max([-min_val, max_val])
        ax.set_ylim([-val, val])
    reset_val = coords[1][0]
    if reset is not None:
        reset_val = reset
    ax.hlines(reset_val, xmin=0, xmax=coords[0][-1],
              linestyle='dashed',
              color='green')
    return fig, ax


class StochasticProcess(ABC):

    def __init__(self, x0):
        self.x0 = x0

    @abstractmethod
    def simulate(self, t, dt):
        pass


class SingleDiffusionProcess(StochasticProcess):

    def __init__(self, x0, D):
        self.x0 = x0
        self.D = D


    def simulate(self, t, dt):
        pos = self.x0
        pos_list = [np.float64(pos)]
        time = np.arange(0, t, dt)
        for step in time[1:]:
            pos += normal_obs(0, 2*self.D) * np.sqrt(dt)
            pos_list.append(pos)
        pos_list = np.array(pos_list)
        return (time, pos_list)

    def plot_simulation(self, t, dt, f1=3.5, f2=2.5):
        coords = self.simulate(t, dt)
        fig, ax = figure_generation_one(coords, f1, f2)
        ax.plot(coords[0], coords[1])


class SingleDiffusionProcessMarkovR(StochasticProcess):

    def __init__(self, x0, xr, D, r):
        self.x0 = x0
        self.xr = xr
        self.D = D
        self.r = r

    def simulate(self, t, dt):
        pos = self.x0
        pos_list = [np.float64(pos)]
        reset_pos = list()
        reset_time = list()
        time = np.arange(0, t, dt)
        for step in time[1:]:
            if np.random.random() < self.r*dt:
                reset_pos.append(pos)
                pos = self.xr
                reset_time.append(step)
            else:
                pos += normal_obs(0, 2*self.D) * np.sqrt(dt)
            pos_list.append(pos)
        return (time, pos_list, reset_time, reset_pos)

    def plot_simulation(self, t, dt, f1=3.5, f2=2.5):
        coords = self.simulate(t, dt)
        fig, ax = figure_generation_one(coords, f1, f2, reset=self.xr)
        ax.plot(coords[0], coords[1])
        if coords[2]:
            for index in range(len(coords[2])):
                ax.vlines(x = coords[2][index],
                           ymin=min(coords[3][index], self.xr),
                           ymax=max(coords[3][index], self.xr),
                           color='r')