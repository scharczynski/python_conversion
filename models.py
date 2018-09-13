import numpy as np
from pyswarm import pso
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns


class Model(object):

    def __init__(self, spikes, time_range, time_bin, x0, bounds):
        self.spikes = spikes   
        print (spikes.shape)
        self.time_range = time_range
        self.time_bin = time_bin
        self.x0 = x0
        self.bounds = bounds
        total_bins  = (time_range[1] - time_range[0]) /  time_bin
        self.t = np.linspace(time_range[0], time_range[1], total_bins)
        self.fit = None
        self.fun = None
        self.lb = [x[0] for x in bounds]
        self.ub = [x[1] for x in bounds]

class Time(Model):

    def __init__(self, spikes, time_range, time_bin, x0, bounds):
        super().__init__(spikes, time_range, time_bin, x0, bounds)
        self.name = "Time"

    def compute_funct(self, x):
        a, ut, st, o = x[0], x[1], x[2], x[3]
        T = a*np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))
        res = np.sum(self.spikes*(-np.log(o+T))+(1-self.spikes)*(-np.log(1-(o+T))))
        # return np.sum(self.spikes*(-np.log(o+T))+(1-self.spikes)*(-np.log(1-(o+T))))
        return res

    def fit_params(self):

        # fits, function_value = pso(self.compute_funct, self.lb, self.ub, maxiter= 1500, f_ieqcons=self.pso_con, debug=True)
        #fits = minimize(self.compute_funct, fits, method='L-BFGS-B', bounds=self.bounds, options={'xtol': 1e-8, 'disp': True})
        fits = minimize(self.compute_funct, self.x0, method='L-BFGS-B', bounds=self.bounds, options={'xtol': 1e-8, 'disp': True})

        # self.fit = fits
        # self.fun = function_value
        self.fit = fits.x
        self.fun = fits.fun

        return fits.x, fits.fun
        # return fits, function_value

    def pso_con(self, x):

        return 1 - (x[0] + x[3])

    def plot_fit(self):

        if self.fit is None:
            print ("fit not yet computed")

        a, ut, st, o = self.fit[0], self.fit[1], self.fit[2], self.fit[3]
        fun = (a*np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o
        plt.subplot(2,1,1)

        plt.plot(self.t, fun)


class Const(Model):

    def __init__(self, spikes, time_range, time_bin, x0, bounds):
        super().__init__(spikes, time_range, time_bin, x0, bounds)
        self.name  = "Constant"

    def compute_funct(self, x):
        o = x[0]
        return np.sum(self.spikes*(-np.log(o))+(1-self.spikes)*(-np.log(1-(o))))

    def fit_params(self):

        fits = minimize(self.compute_funct, self.x0, method='L-BFGS-B', bounds=self.bounds, options={'disp': True})
        self.fit = fits.x
        self.fun = fits.fun

        return fits.x, fits.fun
        
    def plot_fit(self, fit):

        plt.subplot(2,1,1)
        plt.axhline(y=fit, color='r', linestyle='-')

class CatSetTime(Model):

    def __init__(self, spikes, time_params, time_range, time_bin, x0, bounds, conditions):
        super().__init__(spikes, time_range, time_bin, x0, bounds)

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


class CatTime(Model):

    def __init__(self, spikes, time_params, time_range, time_bin, x0, bounds, conditions):
        super().__init__(spikes, time_range, time_bin, x0, bounds)
        self.t = np.tile(self.t, (1064, 1))
        self.c1 = conditions[1][0]
        self.c2 = conditions[2][0]
        self.c3 = conditions[3][0]
        self.c4 = conditions[4][0]
        self.ut = time_params[1]
        self.st = time_params[2]
        self.x0 = x0
        self.bounds = bounds

    def compute_funct(self, x):
        ut, st, o = self.ut, self.st, x[0]
        a1, a2, a3, a4 = x[1], x[2], x[3], x[4]

        big_t = a1 * self.c1 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))) + (
            a2 * self.c2 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))) + (
                a3 * self.c3 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))) + (
                    a4 * self.c4 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))))))

        return np.sum(self.spikes*(-np.log(o+big_t.T))+(1-self.spikes)*(-np.log(1-(o+big_t.T))))

    def fit_params(self):

        # x0 = [0.1,0.1,0.1,0.1,0.1]

        # bound = ((10**-10, 0.2), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2))
        fits = minimize(self.compute_funct, self.x0, method='L-BFGS-B', bounds=self.bounds, options={'disp': True})


        return fits.x, fits.fun

    def plot_fit_full(self, fit):
        ut, st, o = self.ut, self.st, fit[0]
        a1, a2, a3, a4 = fit[1], fit[2], fit[3], fit[4]
        #plt.subplot(2,1,1)
        t = np.linspace(0, 1.6, 1600)
        fun = (a1 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
            a2 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))) + (
                a3 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
                    a4 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))))))) + o


        plt.plot(t, fun)

    def plot_cat_fit(self, fit, cat):
        ut, st, o = self.ut, self.st, fit[0]
        cat_coeff = fit[cat]
        print (cat_coeff)
        t = np.linspace(0, 1.6, 1600)

        fun = (cat_coeff * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.)))) + o
        print(fun)
        plt.plot(t, fun) 

    def plot_c1_fit(self, fit):
        self.plot_cat_fit(fit, 1)

    def plot_c2_fit(self, fit):
        self.plot_cat_fit(fit, 2)

    def plot_c3_fit(self, fit):
        self.plot_cat_fit(fit, 3)

    def plot_c4_fit(self, fit):
        self.plot_cat_fit(fit, 4)

    def plot_all(self, fit):
        
        plt.subplot(2, 2, 1)
        self.plot_c1_fit(fit)

        plt.subplot(2,2,2)
        self.plot_c2_fit(fit)

        plt.subplot(2,2,3)
        self.plot_c3_fit(fit)

        plt.subplot(2, 2, 4)
        self.plot_c4_fit(fit)

    def plot_raster(self, bin_spikes):
        ax = sns.heatmap(bin_spikes)

    def plot_raster_c1(self, bin_spikes):
        ax1 = sns.heatmap(bin_spikes.T * self.c1)

    def plot_raster_c2(self, bin_spikes):
        ax2 = sns.heatmap(bin_spikes.T * self.c2)

    def plot_raster_c3(self, bin_spikes):
        ax3 = sns.heatmap(bin_spikes.T * self.c3)

    def plot_raster_c4(self, bin_spikes):
        ax4 = sns.heatmap(bin_spikes.T * self.c4)
    # def plot_raster_c2(self, bin_spikes):
    #     ax2 = sns.heatmap(bin_spikes * conditions[2][0]

    # def plot_raster_c3(self, bin_spikes):

class ConstCat(Model):

    def __init__(self, spikes, conditions):
        super().__init__(spikes, time_range, time_bin, x0, bounds)

        self.conditions = conditions

        self.c1 = conditions[1][0]
        self.c2 = conditions[2][0]
        self.c3 = conditions[3][0]
        self.c4 = conditions[4][0]

    def compute_funct(self, x):

        a1, a2, a3, a4 = x[0], x[1], x[2], x[3]

        big_t = a1 * self.c1 + a2 * self.c2 + a3 * self.c3 + a4 * self.c4
        print (big_t)
        return np.sum(self.spikes.T*(-np.log(big_t))+(1-self.spikes.T)*(-np.log(1-(big_t))))

    def fit_params(self):

        x0 = [0.01, 0.01, 0.01, 0.01]
        bound = ((10**-10, 0.25), (10**-10, 0.25), (10**-10, 0.25), (10**-10, 0.25))

        fits = minimize(self.compute_funct, x0, method='L-BFGS-B', bounds=bound, options={'disp': True})

        return fits.x, fits.fun

