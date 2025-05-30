import numpy as np
import types
from abc import ABC, abstractmethod
from .__init__ import figure_generation_one, normal_obs


class StochasticProcess(ABC):
    '''
    This defines a parent class for the rest of the processes.
    Mainly for ensuring a simulate function exists.
    '''

    def __init__(self):
        pass

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def update_step(self):
        pass


class StochasticResetting(StochasticProcess):

    def __init__(self):
        super().__init__()


class SingleDiffusionProcess(StochasticProcess):
    '''
    This defines a single particle diffusion process.
    ---
    x0 (float): Initial position of the particle.
    D (float): The diffusion constant of the process.
    '''

    def __init__(self, x0, D):
        self.x0 = x0
        self.D = D

    def update_step(self, pos, dt):
        '''
        Updates the position of the particle based on stochastic rules:
        x(t + dt) = x(t) + sqrt(dt) * N
        for N an observation from a normal random variable with mean 0
        and variance 2*D.
        ---
        pos (float): Position of the particle.
        dt (float): Small change in time used for simulation.
        '''
        return pos + normal_obs(0, 2*self.D) * np.sqrt(dt)

    def simulate(self, t, dt):
        '''
        Simulates the diffusion process over a fixed period of time
        using the stochastic rules:
        x(t + dt) = x(t) + sqrt(dt) * N
        for N an observation from a normal random variable with mean 0
        and variance 2*D.
        ---
        t (float): Time over which the process is simulated.
        dt (float): Small change in time used for simulation.
        '''
        pos = self.x0
        pos_list = [np.float64(pos)]
        time = np.arange(0, t, dt)
        for step in time[1:]:
            pos = self.update_step(pos, dt)
            pos_list.append(pos)
        pos_list = np.array(pos_list)
        return (time, pos_list)

    def plot_simulation(self, t, dt, f1=3.5, f2=2.5):
        '''
        Plots a typical simulation of the single particle diffusion process.
        ---
        t (float): Time over which the process is simulated.
        dt (float): Small change in time used for simulation.

        f1 (float): the length of the figure.
        f2 (float): the height of the figure.
        '''
        coords = self.simulate(t, dt)
        fig, ax = figure_generation_one(coords, f1, f2)
        ax.plot(coords[0], coords[1])

    def first_passage_simulation_constant(self, target, dt=0.1, tmax=100):
        '''
        Returns the first passage time of hitting the target based on
        a simulation of the single particle diffusion process.
        ---
        target (float): Position of target to be reached.
        dt (float): Small change in time used for simulation.
        tmax (float): Maximum time of simulation.
        '''
        if target == self.x0:
            raise ValueError("Target at initial position")
        pos = self.x0
        time = 0
        while time < tmax:
            pos = self.update_step(pos, dt)
            time += dt
            if np.sign(self.x0-target) != np.sign(pos-target):
                return time
        return None

    def mfps(self, target, iter=100, dt=0.1, tmax=100):
        '''
        Returns the sample mean of the first passage time of target
        based on a number of simulations of the single particle
        diffusion process.
        ---
        target (float): Position of target to be reached.
        iter (int): Number of iterations for simulation.
        dt (float): Small change in time used for simulation.
        tmax (float): Maximum time of simulation.
        '''
        count = 0
        total = 0
        for _ in range(iter):
            time = self.first_passage_simulation_constant(target, dt, tmax)
            if time is not None:
                count += 1
                total += time
        if not count:
            raise FirstPassageError('Target was never reached')
        return total/count


class SingleDiffusionProcessConstantR(StochasticResetting):
    '''
    Defines a single particle diffusion process with poissonian resetting.
    ---
    x0 (float): Initial position of the particle.
    xr (float): Reset position.
    D (float): The diffusion constant of the process.
    r (float): Reset rate of the particle.
    '''

    def __init__(self, x0, xr, D, r):
        self.x0 = x0
        self.xr = xr
        self.D = D
        self.r = r

    def update_step(self, pos, dt):
        '''
        Updates the position of the particle based on stochastic rules:
        x(t + dt) = xr  with probability r*dt
        x(t + dt) = x(t) + sqrt(dt) * N  with probability 1-r*dt
        for N an observation from a normal random variable with mean 0
        and variance 2*D.
        ---
        pos (float): Position of the particle.
        dt (float): Small change in time used for simulation.
        '''
        if np.random.random() < self.r*dt:
            return 'reset'
        else:
            return pos + normal_obs(0, 2*self.D) * np.sqrt(dt)

    def simulate(self, t, dt):
        '''
        Simulates the diffusion process with resetting over a fixed period
        of time using the stochastic rules:
        x(t + dt) = xr  with probability r*dt
        x(t + dt) = x(t) + sqrt(dt) * N  with probability 1-r*dt
        for N an observation from a normal random variable with mean 0
        and variance 2*D.
        ---
        t (float): Time over which the process is simulated.
        dt (float): Small change in time used for simulation.
        '''
        pos = self.x0
        pos_list = [np.float64(pos)]
        reset_pos = list()
        reset_time = list()
        time = np.arange(0, t, dt)
        for step in time[1:]:
            pos = self.update_step(pos, dt)
            if pos == 'reset':
                reset_pos.append(pos_list[-1])
                reset_time.append(step)
                pos = self.xr
            pos_list.append(pos)
        return (time, pos_list, reset_time, reset_pos)

    def plot_simulation(self, t, dt, f1=3.5, f2=2.5):
        '''
        Plots a typical simulation of the single particle diffusion process
        with resetting.
        ---
        t (float): Time over which the process is simulated.
        dt (float): Small change in time used for simulation.

        f1 (float): the length of the figure.
        f2 (float): the height of the figure.
        '''
        coords = self.simulate(t, dt)
        fig, ax = figure_generation_one(coords, f1, f2, reset=self.xr)
        ax.plot(coords[0], coords[1])
        if coords[2]:
            # Adding reset lines
            for index in range(len(coords[2])):
                ax.vlines(x=coords[2][index],
                          ymin=min(coords[3][index], self.xr),
                          ymax=max(coords[3][index], self.xr),
                          color='r')

    def first_passage_simulation_constant(self, target, dt=0.1, tmax=100):
        '''
        Returns the first passage time of hitting the target based on
        a simulation of the single particle diffusion process.
        ---
        target (float): Position of target to be reached.
        dt (float): Small change in time used for simulation.
        tmax (float): Maximum time of simulation.
        '''
        time = 0
        pos = self.x0
        cond = self.x0
        while time < tmax:
            pos = self.update_step(pos, dt)
            time += dt
            if pos == 'reset':
                cond = self.xr
                pos = self.xr
            if np.sign(cond-target) != np.sign(pos-target):
                return time
        return None

    def mfpt(self, target, iter=100, dt=0.1, tmax=100):
        '''
        Returns the sample mean of the first passage time of target
        based on a number of simulations of the single particle
        diffusion process with resetting.
        ---
        target (float): Position of target to be reached.
        iter (int): Number of iterations for simulation.
        dt (float): Small change in time used for simulation.
        tmax (float): Maximum time of simulation.
        '''
        count = 0
        total = 0
        for _ in range(iter):
            time = self.first_passage_simulation_constant(target, dt, tmax)
            if time is not None:
                count += 1
                total += time
        if not count:
            raise FirstPassageError('Target was never reached')
        return total/count


class FirstPassageError(ValueError):
    '''Defining an error for first passage time.'''
    pass


class SingleDiffusionProcessResetting(StochasticResetting):
    '''
    Defines a single particle diffusion process with markovian
    but non-poissonian resetting.
    ---
    x0 (float): Initial position of the particle.
    xr (float): Reset position of the particle.
    D (float): Diffusion constant of the process.
    r (function): The density function of the resetting process.
    '''

    def __init__(self, x0, xr, D, r):
        self.x0 = x0
        self.xr = xr
        self.D = D
        self.r = r
        if not isinstance(r, types.FunctionType):
            raise TypeError("r inputed must be a function." +
                            "This can be a constant function.")

    def update_step(self, pos, t_reset, dt):
        '''
        Updates the position of the particle based on stochastic rules:
        x(t + dt) = xr  with probability r(t_reset)*dt
        x(t + dt) = x(t) + sqrt(dt) * N  with probability 1-r(t_reset)*dt
        for N an observation from a normal random variable with mean 0
        and variance 2*D.
        ---
        pos (float): Position of the particle.
        t_reset (float): Time since last reset.
        dt (float): Small change in time used for simulation.
        '''
        if np.random.random() < self.r(t_reset)*dt:
            return 'reset'
        else:
            return pos + normal_obs(0, 2*self.D) * np.sqrt(dt)

    def simulate(self, t, dt):
        '''
        Simulates the diffusion process with resetting over a fixed period
        of time using the stochastic rules:
        x(t + dt) = xr  with probability r(t_reset)*dt
        x(t + dt) = x(t) + sqrt(dt) * N  with probability 1-r(t_reset)*dt
        for N an observation from a normal random variable with mean 0
        and variance 2*D.
        ---
        t (float): Time over which the process is simulated.
        dt (float): Small change in time used for simulation.
        '''
        pos = self.x0
        t_reset = 0
        pos_list = [pos]
        time = np.arange(0, t, dt)
        reset_pos = list()
        reset_time = list()
        for _ in time[1:]:
            pos = self.update_step(pos, t_reset, dt)
            t_reset += dt
            if pos == 'reset':
                reset_pos.append(pos_list[-1])
                reset_time.append(_)
                t_reset = 0
                pos = self.xr
            pos_list.append(pos)
        return (time, pos_list, reset_time, reset_pos)

    def plot_simulation(self, t, dt, f1=3.5, f2=2.5):
        '''
        Plots a typical simulation of the single particle diffusion process
        with non-poissonian but markovian resetting.
        ---
        t (float): Time over which the process is simulated.
        dt (float): Small change in time used for simulation.

        f1 (float): the length of the figure.
        f2 (float): the height of the figure.
        '''
        coords = self.simulate(t, dt)
        fig, ax = figure_generation_one(coords, f1, f2, reset=self.xr)
        ax.plot(coords[0], coords[1])
        if coords[2]:
            # Adding reset lines
            for index in range(len(coords[2])):
                ax.vlines(x=coords[2][index],
                          ymin=min(coords[3][index], self.xr),
                          ymax=max(coords[3][index], self.xr),
                          color='r')

    def first_passage_simulation_constant(self, target, dt=0.1, tmax=100):
        '''
        Returns the first passage time of hitting the target based on
        a simulation of the single particle diffusion process with 
        non-poissonian but markovian resetting.
        ---
        target (float): Position of target to be reached.
        dt (float): Small change in time used for simulation.
        tmax (float): Maximum time of simulation.
        '''
        time = 0
        t_reset = 0
        pos = self.x0
        cond = self.x0
        while time < tmax:
            pos = self.update_step(pos, t_reset, dt)
            time += dt
            t_reset += dt
            if pos == 'reset':
                cond = self.xr
                pos = self.xr
                t_reset = 0
            if np.sign(cond-target) != np.sign(pos-target):
                return time
        return None

    def mfpt(self, target, iter=100, dt=0.1, tmax=100):
        '''
        Returns the sample mean of the first passage time of target
        based on a number of simulations of the single particle
        diffusion process with non-poissonian but markovian resetting.
        ---
        target (float): Position of target to be reached.
        iter (int): Number of iterations for simulation.
        dt (float): Small change in time used for simulation.
        tmax (float): Maximum time of simulation.
        '''
        count = 0
        total = 0
        for _ in range(iter):
            time = self.first_passage_simulation_constant(target, dt, tmax)
            if time is not None:
                count += 1
                total += time
        if not count:
            raise FirstPassageError('Target was never reached')
        return total/count
