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
    binned_spikes : numpy.ndarray
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

        if subsample:
            self.num_trials = int(
                data_processor.num_trials[cell_no] * subsample)

            if self.num_trials < 1:
                self.num_trials = 1

            sampled_trials = np.random.randint(
                data_processor.num_trials[cell_no],
                size=self.num_trials)

            self.binned_spikes = data_processor.binned_spikes[cell_no][sampled_trials, :]
        else:
            self.num_trials = data_processor.num_trials[cell_no]
            self.binned_spikes = data_processor.binned_spikes[cell_no]

        self.cell_no = cell_no
        if data_processor.summed_spikes is not None:
            self.summed_spikes = data_processor.summed_spikes[cell_no]
        else: 
            self.summed_spikes = None

        if data_processor.conditions is not None:
            self.summed_spikes_condition = data_processor.summed_spikes_condition[cell_no]

            conditions_dict = data_processor.conditions_dict

            self.conditions = {}
            for cond in range(1, data_processor.num_conditions + 1):
                self.conditions[cond] = conditions_dict[cond, cell_no]
                if subsample:
                    self.conditions[cond] = self.conditions[cond][sampled_trials]
        else:
            self.conditions = None

        self.time_info = data_processor.time_info

        self.binned_position = data_processor.binned_position

    def fit_model(self, model):
        print(self.cell_no)
        if isinstance(model, models.Const):
            model.fit_params()
        else:
            self.iterate_fits(model, 1)
            #model.fit_params()
            
        return model

   
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
        #fun_min = math.inf
        fun_min = sys.float_info.max
        while iteration < n:
            model.fit_params()
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
        p = chi2.sf(lr, inc_params)
        print(llmin, llmax, inc_params)
        print("p-value is: " + str(p))
        return p < p_threshold

