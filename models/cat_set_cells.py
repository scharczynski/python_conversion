import numpy as np
from pyswarm import pso
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


class CatSetTimeCell(object):

    def __init__(self, spikes, time_params, time_range, time_bin, conditions):

        self.spikes = spikes
        self.t = np.linspace(0, 1.6, 1600)
        self.t = np.tile(self.t, (1064, 1)) 
        self.conditions = conditions
        self.ut = time_params[1]
        self.st = time_params[2]

    def compute_funct(self, x, *args):
        pairs = args
        ut, st, o = self.ut, self.st, x[0]
        a1, a2 = x[1], x[2]
        pair_1 = pairs[0]
        pair_2 = pairs[1]
        c1 = self.conditions[pair_1[0]][0] + self.conditions[pair_1[1]][0]
        c2 = self.conditions[pair_2[0]][0] + self.conditions[pair_2[1]][0]

        big_t = (a1 * c1 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
            a2 * c2 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))))
                

        result = np.sum(self.spikes*(-np.log(o+big_t.T))+(1-self.spikes)*(-np.log(1-(o+big_t.T))))
        return result

    def fit_params(self, pairs):

        x0 = [0.1,0.1,0.1]

        bound = ((10**-10, 0.2), (0, 0.2), (0, 0.2))

        fits = minimize(self.compute_funct, x0, args=(pairs), method='L-BFGS-B', bounds=bound, options={'disp': True})


        return fits.x, fits.fun

    def plot_fit(self, fit):
        ut, st, o = self.ut, self.st, fit[0]
        a1, a2 =  fit[1], fit[2]
        t = np.linspace(0, 1.6, 1600)
        fun = (a1 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
            a2 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))))) + o

        plt.plot(t, fun)

    def plot_raster(self, bin_spikes):
        ax = sns.heatmap(bin_spikes)
