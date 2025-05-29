import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


def plot_simulation(process, t, dt, f1=3.5, f2=2.5):
    if not isinstance(process, StochasticProcess):
        raise TypeError("Can not plot something which isn't a stochastic process")
    coords = process.simulate(t, dt)
    fig, ax = plt.subplots(1, 1, figsize = (f1, f2))
    ax.plot(coords[0], coords[1])
    min_val = np.min(coords[1])
    max_val = np.max(coords[1])
    if min_val >= 0:
        ax.set_ylim([-max_val, max_val])
    elif max_val <= 0:
        ax.set_ylim([min_val, -min_val])
    else:
        val = np.max([-min_val, max_val])
        ax.set_ylim([-val, val])


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

    def normal_obs(self):
        return np.random.normal(loc=0, scale=np.sqrt(2*self.D))

    def simulate(self, t, dt):
        pos = self.x0
        pos_list = [np.float64(pos)]
        time = np.arange(0, t, dt)
        for step in time[1:]:
            pos += self.normal_obs() * np.sqrt(dt)
            pos_list.append(pos)
        pos_list = np.array(pos_list)
        return (time, pos_list)
