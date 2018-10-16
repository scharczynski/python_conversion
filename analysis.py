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
            sampled_trials = np.random.randint(
                data_processor.num_trials[cell_no],
                size=self.num_trials)
            self.binned_spikes = data_processor.binned_spikes[cell_no][sampled_trials, :]
        else:
            self.num_trials = data_processor.num_trials[cell_no]
            self.binned_spikes = data_processor.binned_spikes[cell_no]

        self.cell_no = cell_no
        self.summed_spikes = data_processor.summed_spikes[cell_no]
        self.summed_spikes_condition = data_processor.summed_spikes_condition[cell_no]

        conditions_dict = data_processor.conditions_dict

        # this may be better off as a utility function in python_data
        self.conditions = {}
        for cond in range(1, data_processor.num_conditions + 1):
            self.conditions[cond] = conditions_dict[cond, cell_no]
            if subsample:
                self.conditions[cond] = self.conditions[cond][sampled_trials]
        self.time_info = data_processor.time_info

    def fit_all_models(self):
        return self.fit_time(), self.fit_category_time(), self.fit_constant()

    def fit_time(self):
        n = 2
        mean_delta = 0.10 * (self.time_info.time_high -
                             self.time_info.time_low)
        mean_bounds = (
            (self.time_info.time_low - mean_delta),
            (self.time_info.time_high + mean_delta))
        bounds_t = ((0.001, 1 / n), mean_bounds, (0.01, 5.0), (10**-10, 1 / n))

        m_time = models.Time(
            self.binned_spikes,
            self.time_info,
            bounds_t)

        self.iterate_fits(m_time, 3)
        m_time.update_params()
        self.time_model = m_time
        return m_time

    def fit_constant(self):
        bounds_const = ((10**-10, 0.99),)
        m_constant = models.Const(
            self.binned_spikes,
            self.time_info,
            bounds_const)
        m_constant.fit_params()
        self.constant_model = m_constant
        return m_constant

    def fit_category_time(self):
        bounds_ct = ((10**-10, 0.2), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2))
        m_time_cat = models.CatTime(
            self.binned_spikes,
            self.time_info,
            bounds_ct,
            self.time_model.fit,
            self.conditions,
            self.num_trials
        )
        self.iterate_fits(m_time_cat, 2)
        m_time_cat.update_params()
        self.cat_time_model = m_time_cat
        return m_time_cat

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
        # print (llmin, llmax)
        # print (self.constant_model.fit, self.time_model.fit)
        # print(self.constant_model.fun, self.time_model.fun)
        # print(self.time_model.build_function(self.time_model.fit))
        lr = self.likelihood_ratio(llmin, llmax)
        p = chi2.sf(lr, inc_params)
        print(llmin, llmax, inc_params)
        print("p-value is: " + str(p))
        return p < p_threshold


class AnalyzeAll(object):

    """Performs fitting procedure and model comparison for all cells.

    Parameters
    ----------
    no_cells : int
        Integer signifying the number of cells to be analyzed.
    data_processor : DataProcessor
        Object returned by data processing module that includes all relevent cell data.
    models : list of str
        List of model names as strings to be used in analysis.
        These must match the names used in "fit_x" methods.

    Attributes
    ----------
    no_cells : int
        Integer signifying the number of cells to be analyzed.
    data_processor : DataProcessor
        Object returned by data processing module that includes all relevent cell data.
    models : list of str
        List of model names as strings to be used in analysis.
        These must match the names used in "fit_x" methods.
    time_info : TimeInfo
        Object that holds timing information including the beginning and end of the region
        of interest and the time bin. All in seconds.
    analysis_dict : dict (int: AnalyzeCell)
        Dict containing model fits per cell as contained in AnalyzeCell objects.
    model_fits : dict (str: dict (int: Model))
        Nested dictionary containing all model fits, per model per cell.

    """

    def __init__(self, no_cells, data_processor, models, subsample):
        self.no_cells = no_cells
        self.data_processor = data_processor
        self.time_info = data_processor.time_info
        self.models = models
        self.subsample = subsample
        self.analysis_dict = self.make_analysis()
        self.model_fits = self.fit_all()
        self.subsample = subsample

    def make_analysis(self):
        analysis_dict = {}
        for cell in range(self.no_cells):
            analysis_dict[cell] = AnalyzeCell(
                cell, self.data_processor, self.subsample)
        return analysis_dict

    def fit_all(self):
        model_fits = {}
        for model in self.models:
            model_fits[model] = {} 

        for cell in range(self.no_cells):
            for model in self.models:
                model_fits[model][cell] = getattr(
                    self.analysis_dict[cell], "fit_" + model)()
            # analysis.fit_time()
            # analysis.time_model.plot_fit()
            # print (analysis.time_model.fit)
            # analysis.fit_category_time()
            # analysis.constant_model.plot_fit()
        return model_fits

    def compare_models(self, model_min, model_max):
        for cell in range(self.no_cells):
            plotter = CellPlot(self.analysis_dict[cell])
            min_model = self.model_fits[model_min][cell]
            max_model = self.model_fits[model_max][cell]
            print(
                self.analysis_dict[cell].compare_models(
                    min_model, max_model))
            plotter.plot_comparison(min_model, max_model)
            #self.analysis_dict[cell].plot_cat_fit(max_model)
            # plotter.plot_cat_fit(max_model)
            print(min_model.fit)
            print(max_model.fit)
            plt.show()


path_to_data = '/Users/stevecharczynski/workspace/python_ready_data'
time_info = TimeInfo(0.4, 2.0, 0.001)
data_processor = DataProcessor(path_to_data, time_info, 3, 4)
sns.set()
# analyze_all = AnalyzeAll(1, data_processor, ["time", "category_time"], 0.25)
# analyze_all.compare_models("time", "category_time")
analyze_all = AnalyzeAll(2, data_processor, ["time", "constant"], 0.05)
analyze_all.compare_models("constant", "time")
# analyze_all = AnalyzeAll(2, data_processor, ["time", "category_time"], 0.1)
# analyze_all.compare_models("time", "category_time")
