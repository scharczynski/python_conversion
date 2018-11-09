import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.signal
from models import Const


class CellPlot(object):

    def __init__(self, analysis):
        self.summed_spikes = analysis.summed_spikes
        self.cell_no = analysis.cell_no
        self.num_trials = analysis.num_trials
        self.time_spikes_binned = analysis.time_spikes_binned
        self.conditions = analysis.conditions
        if self.conditions is not None:
            self.summed_spikes_condition = analysis.summed_spikes_condition

        self.time_info = analysis.time_info
        self.subsample = analysis.subsample
        self.t = np.linspace(
            self.time_info.region_low,
            self.time_info.region_high,
            self.time_info.total_bins)

    def plot_raster(self, condition=0):
        if condition:
            ax = sns.heatmap(self.time_spikes_binned.T * self.conditions[condition])
        else:
            ax = sns.heatmap(self.time_spikes_binned.T)

    def plot_cat_fit(self, model):
        fig = plt.figure()
        num_conditions = len(model.conditions.keys())
        fig.suptitle("cell " + str(self.cell_no))

        for condition in model.conditions.keys():
            plt.subplot(2, num_conditions, condition + 1)
            plt.plot(self.t, model.expose_fit(condition), label="fit")
            plt.plot(self.t, self.smooth_spikes(self.summed_spikes_condition[condition]), label="spike_train")
            #plt.plot(self.t, self.smooth_spikes(self.summed_spikes))

        fig_name = "figs/cell_%d_" + model.name + ".png"
        plt.legend(loc="upper left")

        fig.savefig(fig_name % self.cell_no)

    def plot_comparison(self, model_min, model_max):
        fig = plt.figure()
        fig.suptitle("cell " + str(self.cell_no))
        self.plot_fit(model_min)
        plt.plot(self.t, self.smooth_spikes(self.summed_spikes), label="spike_train")
        self.plot_fit(model_max)
        plt.legend(loc="upper left")

        fig_name = "figs/cell_%d_" + model_min.name + "_" + model_max.name + ".png"
        fig.savefig(fig_name % self.cell_no)

    def plot_fit(self, model):
        if isinstance(model, Const):
            plt.axhline(y=model.fit, color='r', linestyle='-')
        else:
            plt.plot(self.t, model.expose_fit(), label=model.name)

    def smooth_spikes(self, spikes):
        if self.subsample:
            avg_spikes = spikes / int(self.num_trials / self.subsample)
        else:
            avg_spikes = spikes / int(self.num_trials)
        # return scipy.signal.savgol_filter(avg_spikes, 251, 3)

        return scipy.ndimage.filters.gaussian_filter(avg_spikes, 50)