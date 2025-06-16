from .standard import StochasticProcess
from .__init__ import normal_obs, exp_obs, figure_generation_one
import matplotlib.pyplot as plt
import numpy as np


accepted_alpha = [1, 2]


class LevyFlight1D(StochasticProcess):

    def __init__(self, alpha, x0, xr, r):
        if alpha not in accepted_alpha:
            raise ValueError(
                "The alpha value provided doesn't have a valid CDF")
        if r < 0 or r > 1:
            raise ValueError("r must satisfy the following: 0<r<1")
        self.x0 = x0
        self.xr = xr
        self.r = r
        self.alpha = alpha

    def obs_value(self, n=None):
        if self.alpha == 1:
            return np.random.standard_cauchy(n)
        else:
            return np.random.normal(0, 1, n)

    def update_step(self, pos):
        if np.random.random() < self.r:
            return 'reset'
        else:
            return pos + self.obs_value()

    def simulate(self, k):
        time = np.arange(0, k, 1)
        pos = self.x0
        pos_list = [pos]
        reset_pos = list()
        reset_time = list()
        for _ in time[1:]:
            pos = self.update_step(pos)
            if pos == 'reset':
                reset_pos.append(pos_list[-1])
                reset_time.append(_)
                pos = self.xr
            pos_list.append(pos)
        return (time, pos_list, reset_time, reset_pos)

    def plot_simulation(self, k, f1=3.5, f2=2.5):
        coords = self.simulate(k)
        fig, ax = figure_generation_one(coords, f1, f2)
        ax.plot(coords[0], coords[1])
        if coords[2]:
            for index in range(len(coords[2])):
                t_val = [coords[2][index]-1, coords[2][index]]
                pos_val = [coords[3][index], self.xr]
                ax.plot(t_val, pos_val, color='r')


class LevyFlight2D(LevyFlight1D):

    def __init__(self, alpha, x0, xr, r):
        if len(x0) != 2 or len(x0) != len(xr):
            raise ValueError("x0 or xr has the wrong length")
        super().__init__(alpha, x0, xr, r)

    def update_step(self, pos):
        if np.random.random() < self.r:
            return 'reset'
        theta = np.random.random() * 2 * np.pi
        dist = self.obs_value()
        return (pos[0]+dist*np.cos(theta), pos[0]+dist*np.sin(theta))

    def simulate(self, k):
        pos = self.x0
        x_val = [self.x0[0]]
        y_val = [self.x0[1]]
        reset_pos_x = list()
        reset_pos_y = list()
        for _ in range(1, k+1):
            pos = self.update_step(pos)
            if pos == 'reset':
                reset_pos_x.append(x_val[-1])
                reset_pos_y.append(y_val[-1])
                pos = self.xr
            x_val.append(pos[0])
            y_val.append(pos[1])
        return (x_val, y_val, reset_pos_x, reset_pos_y)

    def NESS(self, n, k):
        vals = np.full(n, self.x0, dtype=np.float64)
        time = np.arange(0, k, 1)
        for _ in time[1:]:
            cond_vec = np.random.random(n) < self.r
            if len(vals[cond_vec]):
                vals[cond_vec] = self.xr
            if len(vals[cond_vec]) != n:
                vals[np.invert(cond_vec)] = vals[np.invert(cond_vec)]
        return vals

    def plot_simulation(self, k, f=3.5):
        coords = self.simulate(k)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(f, f))
        ax.plot(coords[0], coords[1])
        ax.plot(self.x0[0], self.x0[1], 'o')
        ax.plot(coords[0][-1], coords[1][-1], 'o')
        if coords[2]:
            for _ in range(len(coords[2])):
                x_val = [coords[2][_], self.xr[0]]
                y_val = [coords[3][_], self.xr[1]]
                ax.plot(x_val, y_val, color='r')


class RunAndTumble1D(StochasticProcess):

    def __init__(self, x0, gamma, v0):
        self.x0 = x0
        self.gamma = gamma
        self.v0 = v0

    def update_step(self):
        if np.random.random() < 1/2:
            return 1
        return -1

    def simulate(self, t):
        time = 0
        pos = self.x0
        pos_list = [pos]
        time_list = [time]
        while time < t:
            dir = self.update_step()
            t_step = exp_obs(self.gamma)
            if t_step + time > t:
                pos += dir * self.v0 * (t-time)
                time = t
            else:
                pos += dir * self.v0 * t_step
                time += t_step
            pos_list.append(pos)
            time_list.append(time)
        return (time_list, pos_list)

    def plot_simulation(self, t, f1=3.5, f2=2.5):
        coords = self.simulate(t)
        fig, ax = plt.subplots(1, 1, figsize=(f1, f2))
        fig.set_dpi(600)
        plt.plot(coords[0], coords[1])


class RunAndTumble2D(RunAndTumble1D):

    def __init__(self, x0, xr, gamma, r, v0):
        self.x0 = x0
        self.xr = xr
        self.gamma = gamma
        self.r = r
        self.v0 = v0

    def update_step(self):
        pass

    def simulate(self, t):
        pos = self.x0
        time = 0
        x_list = [pos[0]]
        y_list = [pos[1]]
        reset_x_list = list()
        reset_y_list = list()
        reset_time = exp_obs(self.r)
        dir = np.random.random() * 2 * np.pi
        t_no_reset = 0
        reset=False
        while time < t:
            t_step = exp_obs(self.gamma)
            if reset_time + time > t and t_step + time > t:
                pos += self.v0 * (t-time) * np.array([np.cos(dir), np.sin(dir)])
                time = t
            elif t_no_reset + t_step > reset_time:
                pos += self.v0 * (reset_time-t_no_reset) * np.array([np.cos(dir), np.sin(dir)])
                time += (reset_time - t_no_reset)
                reset_time = exp_obs(self.r)
                t_no_reset = 0
                reset_x_list.append(pos[0])
                reset_y_list.append(pos[1])
                dir = np.random.random() * 2 * np.pi
                reset = True
            else:
                pos += self.v0 * t_step * np.array([np.cos(dir), np.sin(dir)])
                time += t_step
                t_no_reset += t_step
                dir = np.random.random() * 2 * np.pi
            x_list.append(pos[0])
            y_list.append(pos[1])
            if reset == True:
                reset = False
                pos = self.xr
                x_list.append(pos[0])
                y_list.append(pos[1])
        return (x_list, y_list, reset_x_list, reset_y_list)

    def plot_simulation(self, t, f=3.5):
        coords = self.simulate(t)
        fig, ax = plt.subplots(1, 1, figsize=(f, f))
        fig.set_dpi(600)
        ax.plot(coords[0], coords[1])
        ax.plot(self.x0[0], self.x0[1], 'o', label='Initial')
        ax.plot(coords[0][-1], coords[1][-1], 'o', label='Final')
        if coords[2]:
            for _ in range(len(coords[2])):
                x_val = [coords[2][_], self.xr[0]]
                y_val = [coords[3][_], self.xr[1]]
                ax.plot(x_val, y_val, color='r')

    def NESS(self, t, n):
        x_vals = np.full(n, 0)
        y_vals = np.full(n, 0)
        for _ in range(n):
            coords = self.simulate(t)
            x_vals[_] = coords[0][-1]
            y_vals[_] = coords[1][-1]
        return (x_vals, y_vals)

    def plot_NESS(self, t, n, f=3.5, range=None, bins=10):
        coords = self.NESS(t, n)
        fig, ax = plt.subplots(1, 1, figsize=(f, f))
        ax.hist2d(coords[0], coords[1], bins=bins, density=True)
