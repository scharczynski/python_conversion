import numpy as np
from pyswarm import pso
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


class CatTimeCell(object):

    def __init__(self, spikes, time_params, time_range, time_bin, conditions):
        # self.a = params[0]
        # self.ut = params[1]
        # self.st = params[2]
        # self.o = params[3]
        self.spikes = spikes
        # self.lb = lb
        # self.ub = ub
        #self.t = np.arange(time_range[0]+0.001, time_range[1], time_bin)
        self.t = np.linspace(0, 1.6, 1600)
        self.t = np.tile(self.t, (1064, 1))
        
        self.c1 = conditions[1][0]
        self.c2 = conditions[2][0]
        self.c3 = conditions[3][0]
        self.c4 = conditions[4][0]
        self.ut = time_params[1]
        self.st = time_params[2]
        # T = self.a*np.exp((-(t - self.ut)**2)/(2*self.st**2))
        # self.res = np.sum(spikes*(-np.log(self.o+T))+(1-spikes)*(-np.log(1-(self.o+T))))

    def compute_funct(self, x):
        ut, st, o = self.ut, self.st, x[0]
        a1, a2, a3, a4 = x[1], x[2], x[3], x[4]

        # big_t = a1 * np.dot(self.c1 ,np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + (
        #     a2 * np.dot(self.c2 , np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + (
        #         a3 *np.dot(self.c3, np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + (
        #             a4 * np.dot(self.c4, np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))))))

        #with transpose

        big_t = a1 * self.c1 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))) + (
            a2 * self.c2 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))) + (
                a3 * self.c3 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))) + (
                    a4 * self.c4 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))))))

        # big_t = a1 * self.c1 * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.))) + (
        #     a2 * self.c2 * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.))) + (
        #         a3 * self.c3 * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.))) + (
        #             a4 * self.c4 * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.))))))      
        # print (big_t)
        #print (big_t.shape)
        # t1 = a1 * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))
        # t2 = a2 * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))
        # t3 = a3 * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))
        # t4 = a4 * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))
        # big_t = self.c1[0] * t1 + self.c2[0] * t2 + self.c3[0] * t3 + self.c4[0] * t4
        # print(self.c1.shape)
        # print(np.dot(self.c1, self.t))
        result = np.sum(self.spikes*(-np.log(o+big_t.T))+(1-self.spikes)*(-np.log(1-(o+big_t.T))))
        # result = np.sum(self.spikes*(-np.log(o+big_t))+(1-self.spikes)*(-np.log(1-(o+big_t))))
        return result
        # print (result)
        # print (result.shape)
        # return np.sum(self.spikes*(-np.log(o+big_t))+(1-self.spikes)*(-np.log(1-(o+big_t))))
        # return np.sum(self.spikes*(-np.log(o+big_t.T))+(1-self.spikes)*(-np.log(1-(o+big_t.T))))

    def fit_params(self):

        x0 = [0.1,0.1,0.1,0.1,0.1]

        bound = ((10**-10, 0.2), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2))
        # fits, function_value = pso(self.compute_funct, self.lb, self.ub, maxiter=1400)
        # fits = minimize(self.compute_funct, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
        fits = minimize(self.compute_funct, x0, method='L-BFGS-B', bounds=bound, options={'disp': True})


        return fits.x, fits.fun
        #return fits, function_value

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
