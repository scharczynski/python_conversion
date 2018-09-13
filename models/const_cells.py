import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class ConstantCell(object):

    def __init__(self, spikes, time_range, time_bin):

        self.spikes = spikes
        total_bins  = (time_range[1] - time_range[0]) /  time_bin
        self.t = np.linspace(time_range[0], time_range[1], total_bins) #given time bin in milliseconds 
        self.fit = None
        self.fun = None

    def compute_funct(self, x):
        o = x[0]
        return np.sum(self.spikes*(-np.log(o))+(1-self.spikes)*(-np.log(1-(o))))

    def fit_params(self):
        x0 = [0.1]
        bound = ((10**-10, 1),)

        fits = minimize(self.compute_funct, x0, method='L-BFGS-B', bounds=bound, options={'disp': True})
        return fits.x, fits.fun
        
    def plot_fit(self, fit):

        plt.subplot(2,1,1)
        plt.axhline(y=fit, color='r', linestyle='-')
