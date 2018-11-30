import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.signal
from models import Const


class CellPlot(object):

    def __init__(self, analysis):
        # self.time_spikes_summed = analysis.time_spikes_summed
        # self.position_spikes_summed = analysis.position_spikes_summed
        self.analysis = analysis
        self.cell_no = analysis.cell_no
        self.num_trials = analysis.num_trials
        # self.time_spikes_binned = analysis.time_spikes_binned
        self.conditions = analysis.conditions
        # if self.conditions is not None:
        #     self.time_spikes_summed_condition = analysis.time_spikes_summed_cat
        #     self.position_spikes_summed_cat = analysis.position_spikes_summed_cat
        self.time_info = analysis.time_info
        self.pos_info = analysis.pos_info
        self.subsample = analysis.subsample
        # self.t = np.linspace(
        #     self.time_info.region_low/1000,
        #     self.time_info.region_high/1000,
        #     self.time_info.total_bins)
        # self.x = np.linspace(self.pos_info.region_low, self.pos_info.region_high, self.pos_info.total_bins)

    def plot_raster(self, condition=0):
        cmap = sns.cm.rocket_r
        if condition:
            ax = sns.heatmap(self.time_spikes_binned.T * self.conditions[condition])
        else:
            ax = sns.heatmap(self.analysis.time_spikes_binned, cmap=cmap)

    def plot_cat_fit(self, model):
        fig = plt.figure()
        num_conditions = len(model.conditions.keys())
        fig.suptitle("cell " + str(self.cell_no))

        for condition in model.conditions.keys():
            plt.subplot(2, num_conditions, condition + 1)
            plt.plot(model.region, model.expose_fit(condition), label="fit")
            plt.plot(model.region, self.smooth_spikes(self.get_model_sum(model, True)[condition]), label="spike_train")
            #plt.plot(self.t, self.smooth_spikes(self.summed_spikes))

        fig_name = "figs/cell_%d_" + model.name + ".png"
        plt.legend(loc="upper left")

        fig.savefig(fig_name % self.cell_no)

    def plot_comparison(self, model_min, model_max):
        fig = plt.figure()
        fig.suptitle("cell " + str(self.cell_no))
        plt.subplot(2,1,1)
        self.plot_fit(model_min)
        #plt.plot(self.t, self.smooth_spikes(self.time_spikes_summed), label="spike_train")
        plt.plot(model_max.region, self.smooth_spikes(self.get_model_sum(model_max)), label="spike_train")
        self.plot_fit(model_max)
        plt.legend(loc="upper left")
        plt.subplot(2,1,2)
        self.plot_raster()
        fig_name = "figs/cell_%d_" + model_min.name + "_" + model_max.name + ".png"
        fig.savefig(fig_name % self.cell_no)

    def plot_fit(self, model):
        if isinstance(model, Const):
            plt.axhline(y=model.fit, color='r', linestyle='-')
        else:
            plt.plot(model.region, model.expose_fit(), label=model.name)
            # plt.plot(self.x, model.expose_fit(), label=model.name)


    def smooth_spikes(self, spikes):
        if self.subsample:
            avg_spikes = spikes / int(self.num_trials / self.subsample)
        else:
            avg_spikes = spikes / int(self.num_trials)
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
            return self.analysis.time_spikes_summed


