import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.signal
from models import Const


class CellPlot(object):

    def __init__(self, analysis):
        self.summed_spikes = analysis.summed_spikes
        self.summed_spikes_condition = analysis.summed_spikes_condition
        self.cell_no = analysis.cell_no
        self.num_trials = analysis.num_trials
        self.binned_spikes = analysis.binned_spikes
        self.conditions = analysis.conditions
        self.time_info = analysis.time_info
        self.subsample = analysis.subsample
        self.t = np.linspace(self.time_info.time_low, self.time_info.time_high, self.time_info.total_bins)

    def plot_raster(self, condition=0):
        if condition:
            ax = sns.heatmap(self.binned_spikes.T * self.conditions[condition])
        else:
            ax = sns.heatmap(self.binned_spikes.T)

    def plot_cat_fit(self, model):
        ut, st, o = model.ut, model.st, model.o
        num_conditions = len(model.conditions.keys())
        for condition in range(num_conditions):
            plt.subplot(2, num_conditions, condition+1)
            plt.plot(self.t, (model.fit[condition] * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))))
            plt.plot(self.t, scipy.signal.savgol_filter((self.summed_spikes_condition[condition])/int(self.num_trials/self.subsample), 251, 3))

    def plot_comparison(self, model_min, model_max):
        fig = plt.figure()
        self.plot_fit(model_min)
        plt.plot(self.t, scipy.signal.savgol_filter((self.summed_spikes/int(self.num_trials/self.subsample)), 251, 3))
        self.plot_fit(model_max)
        
        fig_name = "figs/cell_%d_" + model_min.name + "_" + model_max.name + ".png"
        fig.savefig(fig_name % self.cell_no)

    def plot_fit(self, model):
        if type(model) is Const:
            plt.axhline(y=model.fit, color='r', linestyle='-')
        else:    
            plt.plot(self.t, model.expose_fit())
    