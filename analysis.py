"""Module containing classes that perform maximum likelihood analysis.

    Model fitting and statistic calculation are performed in AnalyzeCell.
    AnalyzeAll extends the analysis to all provided cells and allows comparison of
    arbitrary nested models.


"""

import numpy as np
from python_data import DataProcessor
import matplotlib.pyplot as plt
from scipy.stats import chi2
import seaborn as sns
import scipy.signal
import models
import math
from time_info import TimeInfo
import sys
from cellplot import CellPlot
import time
import scipy.signal


class AnalyzeCell(object):

    """Performs fitting procedure and model comparison for one cell.

    Parameters
    ----------
    cell_no : int
        Integer signifying the cell being analyzed.
    data_processor : DataProcessor
        Object returned by data processing module that includes all relevent cell data.

    Attributes
    ----------
    summed_spikes : numpy.ndarray
        Array containing summed spike data for given cell.
    time_spikes_binned : numpy.ndarray
        Array containing binned spike data or given cell.
    conditions : dict (int: numpy.ndarray of int)
        Dict containing condition information for the cell.
    num_trials : int
        Int signifying the number of trials for given cell.
    time_info : TimeInfo
        Object that holds timing information including the beginning and end of the region
        of interest and the time bin. All in seconds.

    """

    def __init__(self, cell_no, data_processor, subsample):
        self.subsample = subsample
        self.time_info = data_processor.data_descriptor.time_info
        self.pos_info = data_processor.data_descriptor.pos_info
        self.cell_no = cell_no

        #if subsampling is applied, random trials are selected
        if subsample:
            sampled_trials = self.subsample_trials(data_processor.num_trials[cell_no], subsample)
            self.num_trials = len(sampled_trials)
        else:
            self.num_trials = data_processor.num_trials[cell_no]

        #gathering various spike data if available
        if data_processor.data_descriptor.pos_info is not None:
            if subsample:
                self.position_spikes_binned = self.apply_subsample(
                    data_processor.position_spikes_binned[cell_no],
                    sampled_trials
                )
            else:
                self.position_spikes_binned = data_processor.position_spikes_binned[cell_no]
            self.position_spikes_summed = data_processor.position_spikes_summed[cell_no]
            if data_processor.data_descriptor.num_conditions:
                self.position_spikes_summed_cat = data_processor.position_spikes_summed_cat[cell_no]
        else:
            self.position_spikes_binned = None
            self.position_spikes_summed = None       

        if data_processor.data_descriptor.time_info is not None:
            if subsample:
                self.time_spikes_binned = self.apply_subsample(
                    data_processor.time_spikes_binned[cell_no],
                    sampled_trials
            )
            else:
                self.time_spikes_binned = data_processor.time_spikes_binned[cell_no]
            self.time_spikes_summed = data_processor.time_spikes_summed[cell_no]
            #test if this condition is required
            if data_processor.data_descriptor.num_conditions:
                self.time_spikes_summed_cat = data_processor.time_spikes_summed_cat[cell_no]
        else:
            self.time_spikes_binned = None
            self.time_spikes_summed = None

        #transform condition dictionary into required format
        conditions_dict = data_processor.conditions_dict

        if data_processor.num_conditions:
            self.conditions = {}
            for cond in range(1, data_processor.num_conditions + 1):
                self.conditions[cond] = conditions_dict[cond, cell_no]
                if subsample:
                    self.conditions[cond] = self.conditions[cond][sampled_trials]
        else:
            self.conditions = None


    def apply_subsample(self, spikes, sampled_trials):
        return spikes[sampled_trials, :]
        
    def subsample_trials(self, num_trials, subsample):
        num_trials = int(
            num_trials * subsample)

        if num_trials < 1:
            num_trials = 1

        sampled_trials = np.random.randint(
            num_trials,
            size=num_trials)

        return sampled_trials


    def fit_model(self, model):
        print(self.cell_no)

        #constant models at some point got stuck in "improvement" loop
        if isinstance(model, models.Const):
            model.fit_params()
        else:
            self.iterate_fits(model, 10)
        if model.name is "time":
            self.save_fit_params(model)
        
        return model

    def save_fit_params(self, model):
        np.save(("/usr3/bustaff/scharcz/workspace/fit_results/cell_" + str(self.cell_no) + "_" + model.name + "_results"), model.fit)
   
    def compare_models(self, model_min, model_max):
        llmin = model_min.fun
        llmax = model_max.fun
        delta_params = model_max.num_params - model_min.num_params
        return self.hyp_rejection(
            0.05,
            llmin,
            llmax,
            delta_params)

    def iterate_fits(self, model, n):
        iteration = 0
        fun_min = sys.float_info.max
        while iteration < n:
            model.fit_params()
            #check if the returned fit is better by at least a tiny amount
            if model.fun < (fun_min - fun_min * 0.0001):
                fun_min = model.fun
                params_min = model.fit
                iteration = 0
            else:
                iteration += 1
        model.fit = params_min
        model.fun = fun_min
        return model

    def likelihood_ratio(self, llmin, llmax):
        return(-2 * (llmax - llmin))

    def hyp_rejection(self, p_threshold, llmin, llmax, inc_params):
        lr = self.likelihood_ratio(llmin, llmax)
        #this test was chosen somewhat arbitrarily, sf is the survival function
        p = chi2.sf(lr, inc_params)
        print(llmin, llmax, inc_params)
        print("p-value is: " + str(p))
        return p < p_threshold

