import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.signal
from models import Const
import matplotlib as mpl


class CellPlot(object):

    def __init__(self, analysis):
        self.analysis = analysis
        self.cell_no = analysis.cell_no
        self.num_trials = analysis.num_trials
        self.conditions = analysis.conditions
        self.time_info = analysis.time_info
        self.pos_info = analysis.pos_info
        self.subsample = analysis.subsample

    def plot_raster(self, condition=0):
        if condition:
            scatter_data = np.nonzero(self.analysis.time_spikes_binned.T * self.conditions[condition])
        else:
            scatter_data = np.add(np.nonzero(self.analysis.time_spikes_binned.T), self.time_info.region_low)

        plt.scatter(scatter_data[0], scatter_data[1], c=[[0,0,0]], marker="o", s=1)

    def plot_cat_fit(self, model):
        fig = plt.figure()    
        num_conditions = len(model.conditions)
        fig.suptitle("cell " + str(self.cell_no))
        fig_name = "figs/cell_%d_" + model.name + ".png"
        plt.legend(loc="upper left")

        for condition in model.conditions:
            plt.subplot(2, num_conditions, condition + 1)
            plt.plot(model.region, model.expose_fit(condition), label="fit")
            plt.plot(model.region, self.smooth_spikes(self.get_model_sum(model, True)[condition]), label="spike_train")

        fig.savefig(fig_name % self.cell_no)

    def plot_comparison(self, model_min, model_max):
        fig = plt.figure()
        fig.suptitle("cell " + str(self.cell_no))
        fig_name = "figs/cell_%d_" + model_min.name + "_" + model_max.name + ".png"
        plt.subplot(2,1,1)
        self.plot_fit(model_min)
        plt.plot(model_max.region, self.smooth_spikes(self.get_model_sum(model_max)), label="spike_train")
        self.plot_fit(model_max)
        plt.legend(loc="upper right")

        plt.subplot(2,1,2)
        self.plot_raster()

        fig.savefig(fig_name % self.cell_no)

    def plot_fit(self, model):
        if isinstance(model, Const):
            plt.axhline(y=model.fit, color='r', linestyle='-')
        else:
            plt.plot(model.region, model.expose_fit(), label=model.name)

    def smooth_spikes(self, spikes):
        if self.subsample:
            avg_spikes = spikes / int(self.num_trials / self.subsample)
        else:
            avg_spikes = spikes / int(self.num_trials)
            # avg_spikes = spikes / 2000

        # return scipy.signal.savgol_filter(avg_spikes, 251, 3)
        return scipy.ndimage.filters.gaussian_filter(avg_spikes, 50)

    def get_model_sum(self, model, cat=0):
        if model.model_type == "position":
            if cat:
                return self.analysis.position_spikes_summed_cat
            else:
                return self.analysis.position_spikes_summed
        elif model.model_type == "time":
            if cat:
                return self.analysis.time_spikes_summed_cat
            else:
                return self.analysis.time_spikes_summed


