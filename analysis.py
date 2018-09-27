import numpy as np
import parse_matlab_data
import python_data
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarm import pso
import scipy.ndimage
from scipy.stats import chi2
import seaborn as sns
import scipy.signal
import models

class TimeInfo(object):
    def __init__(self, time_low, time_high, time_bin):
        self.time_low = time_low
        self.time_high = time_high
        self.time_bin = time_bin

class AnalyzeCell(object):

    def __init__(self, cell_no, data, time_info):
        self.cell_no = cell_no
        print(data[0][0].shape)
        print(data[1][0].shape)
        self.summed_spikes = data[0][cell_no]
        self.binned_spikes = data[1][cell_no]
        self.conditions_dict = data[2]
        self.time_info = time_info

    def fit_all_models(self):
        return self.fit_time(), self.fit_category_time(), self.fit_constant()
 

    def fit_time(self):
        n = 2
        mean_delta = 0.10 * (self.time_info.time_high - self.time_info.time_low)
        mean_bounds = ((self.time_info.time_low - mean_delta), (self.time_info.time_high + mean_delta))
        bounds_t = ( (0.001, 1/n), mean_bounds, (0.01, 5.0), (10**-10, 1/n))

        m_time = models.Time(
            self.cell_no, 
            self.binned_spikes, 
            self.time_info,
            bounds_t)

        self.iterate_fits(m_time, 3)
        self.time_model = m_time
        return m_time

    def fit_constant(self):
        bounds_const = ((10**-10, 0.99),)
        m_constant = models.Const(
            self.cell_no, 
            self.binned_spikes, 
            self.time_info,
            bounds_const)
        #self.iterate_fits(m_constant, 1)
        m_constant.fit_params()
        self.constant_model = m_constant
        return m_constant

    def fit_category_time(self):
        bounds_ct = ((10**-10, 0.2), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2))
        m_time_cat = models.CatTime(
            self.cell_no, 
            self.binned_spikes,  
            self.time_info,
            bounds_ct, 
            self.time_model.fit,
            self.conditions_dict
        )
        self.iterate_fits(m_time_cat, 1)
        return m_time_cat

    def compare_models(self, model_min, model_max, delta_params):
        llmin = model_min.fun
        llmax = model_max.fun
        return self.hyp_rejection(
                                0.05,
                                llmin,
                                llmax, 
                                delta_params)

    def plot_comparison(self, model_min, model_max):
        fig = plt.figure()
        model_min.plot_fit()
        plt.plot(np.linspace(0.4, 2.0, 1600), scipy.signal.savgol_filter((self.summed_spikes/1000), 153, 3))
        model_max.plot_fit()
        print(model_max.fit, model_min.fit)
        fig_name = "figs/cell_%d.png" % self.cell_no
        fig.savefig(fig_name)

    def iterate_fits(self, model, n):
        iteration = 0
        fun_min = math.inf
        while iteration < n:
            print(model.fit, model.fun)
            model.fit_params()
            print(model.fit, model.fun)
            if model.fun < fun_min:
                fun_min = model.fun
                params_min = model.fit
                iteration = 0
            else:
                iteration += 1  
        model.fit = params_min
        model.fun = fun_min
        return model

    def likelihood_ratio(self, llmin, llmax):
        return(-2*(llmax-llmin))

    def hyp_rejection(self, p_threshold, llmin, llmax, inc_params):
        print (llmin, llmax)
        print (self.constant_model.fit, self.time_model.fit)
        print(self.constant_model.fun, self.time_model.fun)
        print(self.time_model.build_function(self.time_model.fit))
        lr = self.likelihood_ratio(llmin, llmax)
        p = chi2.sf(lr, inc_params)
        print ("p-value is: " + str(p))
        return p < p_threshold

class AnalyzeAll(object):

    def __init__(self, no_cells, data, time_info):
        self.no_cells = no_cells
        self.data = data
        self.time_info = time_info

    def compare_all(self):
        for cell in range(self.no_cells):
            analysis = AnalyzeCell(cell, self.data, self.time_info)
            analysis.fit_time()
            # analysis.time_model.plot_fit()
            # print (analysis.time_model.fit)
            analysis.fit_constant()
            # analysis.constant_model.plot_fit()
            
            print(analysis.compare_models(analysis.constant_model, analysis.time_model, 3))
            analysis.plot_comparison(analysis.constant_model, analysis.time_model)




data = python_data.parse_python(3, [0.4, 2.0], 0.001)

# summed_spikes = data[0]

# plt.plot(summed_spikes[0])
# plt.show()
time_info = TimeInfo(0.4, 2.0, 0.001)
sns.set()
analyze_all = AnalyzeAll(1, data, time_info)
analyze_all.compare_all()

# fun_min = math.inf
# fun_min_cat = math.inf
# c = 0
# c1 = 0
# for i in range(1):
#     analysis = AnalyzeCell(i, data, time_info)
#     model = analysis.fit_time()
#     # print(analysis.cell_no)
#     while c < 2:
#         model = analysis.fit_time()
#         print(model.fit)
#         if model.fun < fun_min:
#             fun_min = model.fun
#             params_min = model.fit
#             c = 0
#         else:
#             c += 1
#     while c1 < 2:
#         model_cat_time = analysis.fit_category_time()
#         if model_cat_time.fun < fun_min_cat:
#             fun_min_cat = model_cat_time
#             params_min_cat = model_cat_time.fit
#             c1 = 0
#         else:
#             c1 += 1
    
    
    # fig = plt.figure()
    # model.plot_fit()
    # plt.plot(np.linspace(0.4, 2.0, 1600), scipy.signal.savgol_filter((analysis.summed_spikes/1000), 153, 3))
    # model_cat_time.plot_fit_full()
    # print(model_cat_time.fit, model.fit)
    # fig_name = "figs/cat_cell_%d.png" % i 
    # fig.savefig(fig_name)
    # print(analysis.hyp_rejection(0.05, model.fun, model_cat_time.fun, 3))


# analysis_0 = AnalyzeCell(0, data, time_range, time_bin)
# m_time = analysis_0.fit_time()
# m_time.plot_fit()
# plt.plot(np.linspace(0.4, 2.0, 1600), scipy.signal.savgol_filter((analysis_0.summed_spikes/1000), 153, 3))
# plt.subplot(2,1,2)
# plt.plot(analysis_0.summed_spikes)

# m_const = analysis_0.fit_constant() 
# m_cat_time = analysis_0.fit_category_time()
# plt.subplot(3,1,1)
# m_time.plot_fit()
# plt.subplot(3,1,2)
# m_cat_time.plot_fit_full()
# plt.subplot(3,1,3)
# plt.plot(analysis_0.summed_spikes)
# plt.show()
# print(analysis_0.compare_models(m_const, m_time, 3))
# print(analysis_0.compare_models(m_time, m_cat_time, 3))
# analysis_1 = AnalyzeCell(1, data, time_range, time_bin)
# analysis_2 = AnalyzeCell(2, data, time_range, time_bin)
# # analysis_3 = AnalyzeCell(3, data, time_range, time_bin)
# x0_t = [0.1, 0.5, 0.5, 0.01]
# bounds_t = ( (0.001, 0.2), (0.1, 0.9), (0.1, 5.0), (10**-10, 0.2))
# plt.subplot(2,1,1)
# m_time_0 = models.Time(analysis_0.binned_spikes, analysis_0.time_range, analysis_0.time_bin, x0_t, bounds_t)
# m_time_0.fit_params()
# m_time_0.plot_fit()
# plt.subplot(2,1,2)
# pso = models.Time(analysis_0.binned_spikes, analysis_0.time_range, analysis_0.time_bin, x0_t, bounds_t)
# pso.fit_params_pso()
# pso.plot_fit()

# plt.show()


# m_time_1 = models.Time(analysis_1.binned_spikes, analysis_1.time_range, analysis_1.time_bin, x0_t, bounds_t)
# m_time_1.fit_params()
# m_time_1.plot_fit()
# plt.subplot(2,2,1)
# m_time_1 = models.Time(analysis_1.binned_spikes, analysis_1.time_range, analysis_1.time_bin, x0_t, bounds_t)
# m_time_1.fit_params_pso()
# m_time_1.plot_fit()
# plt.subplot(2,2,2)
# not_pso = models.Time(analysis_1.binned_spikes, analysis_1.time_range, analysis_1.time_bin, x0_t, bounds_t)
# not_pso.fit_params()
# not_pso.plot_fit()
# plt.show()

# t = np.linspace(1, 1.572, 1572)
# # t = np.linspace(0, 1.479, 1479)
# # t = np.linspace(0, 1.599, 1599)

# plt.plot(t, analysis_0.summed_spikes)
# print(m_time_0.fun, pso.fun)
# # plt.plot(t, analysis_0.summed_spikes)
# plt.show()

# t = np.linspace(0,1.572, 1572)
# plt.subplot(2,1,2)
# plt.plot(t, analysis.summed_spikes)
# plt.show()
# print (m_time.fit)

# x0_const = [0.1]
# bounds_const = ((10**-10, 1),)
# m_constant = models.Const(analysis.binned_spikes, analysis.time_range, analysis.time_bin, x0_const, bounds_const)

# x0_ct = [0.1,0.1,0.1,0.1,0.1]
# bounds_ct = ((10**-10, 0.2), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2))
# print (m_time.fit)
# m_time_cat = models.CatTime(analysis.binned_spikes, m_time.fit, analysis.time_range, analysis.time_bin, x0_ct, bounds_ct, data[2])
# # m_time_cat.fits, m_time_cat.fun = m_time_cat.fit_params()

# fit_outcome = analysis.compare_models(m_time, m_time_cat, 0.05, 3)


# print (fit_outcome)


