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
import models


class AnalyzeCell(object):

    def __init__(self, cell_no, data, time_range, time_bin):
        self.summed_spikes = data[0][cell_no]
        self.binned_spikes = data[1][cell_no]
        self.conditions_dict = data[2]
        self.time_range = time_range
        self.time_bin = time_bin

    def compare_models(self, model_min, model_max, p_threshold, delta_params):

        fits_min, fun_min = model_min.fit_params()
        fits_max, fun_max = model_max.fit_params()

        return self.hyp_rejection(p_threshold, fun_min, fun_max, delta_params)

    def likelihood_ratio(self, llmin, llmax):
        return(-2*(llmax-llmin))

    def hyp_rejection(self, p_threshold, llmin, llmax, inc_params):
        lr = self.likelihood_ratio(llmin, llmax)
        p = chi2.sf(lr, inc_params)
        print ("p-value is: " + str(p))
        return p < p_threshold

data = python_data.parse_python(2, [0, 1.6], 0.001)

# summed_spikes = data[0]

# plt.plot(summed_spikes[0])
# plt.show()


time_range = (0, 1.6)
time_bin = 0.001

analysis = AnalyzeCell(0, data, time_range, time_bin)

x0_t = [0.1, 0.5, 0.5, 0.01]
bounds_t = ( (0.001, 0.2), (0.1, 0.9), (0.1, 5.0), (10**-10, 5))
m_time = models.Time(analysis.binned_spikes, analysis.time_range, analysis.time_bin, x0_t, bounds_t)
m_time.fit_params()
m_time.plot_fit()
plt.subplot(2,1,2)
t = np.linspace(0, 1.572, 1572)
plt.plot(t, analysis.summed_spikes)
plt.show()

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


