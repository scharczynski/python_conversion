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
from models import cat_set_cells, cat_time_cells, time_cells, const_cat_cells, const_cells
from scipy.stats import chi2



#spikes, conditions, num_trials, trial_length, summed_spikes, binned_spikes  = parse_matlab_data.import_data()

summed_spikes, binned_spikes, conditions_dict = python_data.parse_python()



cell_spikes = binned_spikes
#cell_spikes = binned_spikes[0]








# bounds = ((0,0.5), (0.01, 5), (-0.9, 0.9), (10**-10, 0.5))
# params = [0.01, 0.02 ,-0.7, 0.001]
# params2 = [-0.00711966,  0.01972802,  0.02054082,  0.01048906]


# fun_min = math.inf
cell_0 = time_cells.TimeCell(binned_spikes, (0, 1.6), 0.001)
fits_t, fun_t = cell_0.fit_params()
# cell_0.plot_fit(fits_t)

# # lb_c = [0, 0.1, 10**-10, 0.001, 0.001, 0.001, 0.001]
# # ub_c = [0.9, 5.0, 0.2, 0.2, 0.2, 0.2, 0.2]


# cell_0_cat = cat_time_cells.CatTimeCell(binned_spikes, fits_t, (0, 1.6), 0.001, conditions_dict)
# fits_cat, fun_cat = cell_0_cat.fit_params()
# # cell_0_cat.plot_fit_full(fits_cat)
# # print(fits_cat)

# plt.subplot(2,1,1)
# cell_0_cat.plot_c1_fit(fits_cat)


# cell_0_cat_set = cat_set_cells.CatSetTimeCell(binned_spikes, fits_t, (0, 1.6), 0.001, conditions_dict)
# fits_set, fun_set = cell_0_cat_set.fit_params(((1,2), (3,4)))
#cell_0_cat_set.plot_fit(fits_set)
# plt.show()

# cell_0_cat.plot_c2_fit(fits_cat)
# plt.show()

# cell_0_cat.plot_c3_fit(fits_cat)
# plt.show()

# cell_0_cat.plot_c4_fit(fits_cat)
# plt.show()
# cell_0_cat.plot_all(fits_cat)

cell_0_const = const_cells.ConstantCell(binned_spikes, (0, 1.6), 0.001)
fits_c, fun_c = cell_0_const.fit_params()

cell_0_const_cat = const_cat_cells.ConstCatCell(binned_spikes, conditions_dict)
fits_c_cat, fun_c_cat = cell_0_const_cat.fit_params()
print (fits_c)
print (fits_c_cat)
# print(fun_t, fun_cat)

def likelihood_ratio(llmin, llmax):
    return(-2*(llmax-llmin))


lr = likelihood_ratio(fun_c, fun_c_cat)

p = chi2.sf(lr, 3)

print (lr)
print (p)

#rasters----------

# plt.subplot(2,3,1)
# cell_0_cat.plot_raster(binned_spikes)
# plt.subplot(2,3,2)
# cell_0_cat.plot_raster_c1(binned_spikes)
# plt.subplot(2,3,3)
# cell_0_cat.plot_raster_c2(binned_spikes)
# plt.subplot(2,3,4)
# cell_0_cat.plot_raster_c3(binned_spikes)
# plt.subplot(2, 3,5)
# cell_0_cat.plot_raster_c4(binned_spikes)


# ut, st, o = fits[0], fits[1], fits[2]
# a1, a2, a3, a4 = fits[3], fits[4], fits[5], fits[6]
# t = np.arange(0, 1.6, 0.001)
# #t = np.tile(t, (1064, 1))
# c1, c2, c3, c4 = conditions_dict[1][0], conditions_dict[2][0], conditions_dict[3][0], conditions_dict[4][0]

# print(a1,a2,a3,a4)
# print(ut,st,o)
# plt.subplot(2,1,1)
# # plt.plot(a1 * np.dot(c1, np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.)))) + (
# #     a2 * np.dot(c2, np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.)))) + (
# #             a3 * np.dot(c3, np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.)))) + (
# #                 a4 * np.dot(c4, np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))))))))
# plt.plot(a1 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
#     a2 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))) + (
#             a3 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
#                 a4 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.)))))))
# while c < 3:
#     fits, fun = cell_0.fit_params()

#     if fun <= fun_min:
#         fun_min = fun
#         params_min = fits
#         c = 0
#     else:
#         c += 1
    


# a0 = params_min[0]
# ut0 = params_min[1]
# st0 = params_min[2]
# o0 = params_min[3]
# t = np.linspace(0, 1.6, 1600)
# print(fits)
# plt.subplot(2,1,1)

# plt.plot(a0*np.exp(-np.power(t - ut0, 2.) / (2 * np.power(st0, 2.))))

#xopt, fopt = pso(swarm_fun, lb, ub)
#print (xopt)

# c_xopt, c_fopt = pso(constant, [10**-10], [0.5])

# t = np.arange(0, 1.6, 0.001)
# #plt.plot(summed_spikes[0])
# a0 = xopt[0]
# ut0 = xopt[1]
# st0 = xopt[2]
# o0 = xopt[3]

# x0 = np.array([a0, ut0, st0, o0])

# plt.plot(a0*np.exp(-np.power(t - ut0, 2.) / (2 * np.power(st0, 2.))))

# plt.subplot(2,1,3)
t2 = np.linspace(0, 1.572, 1572)
t = np.linspace(0,1.6, 1600)

# plt.plot(t, summed_spikes)

spikes_c1 = binned_spikes.T * conditions_dict[1][0]


sum_c1 = np.zeros(1600)
for i in range(spikes_c1.shape[0]):
    for j in spikes_c1[i]:
        if j == 1:
            sum_c1[i] += 1

spikes_c2 = binned_spikes.T * conditions_dict[2][0]

sum_c2 = np.zeros(1600)
for i in range(spikes_c2.shape[0]):
    for j in spikes_c2[i]:
        if j == 1:
            sum_c2[i] += 1

spikes_c3 = binned_spikes.T * conditions_dict[3][0]

sum_c3 = np.zeros(1600)
for i in range(spikes_c3.shape[0]):
    for j in spikes_c3[i]:
        if j == 1:
            sum_c3[i] += 1

spikes_c4 = binned_spikes.T * conditions_dict[4][0]

sum_c4 = np.zeros(1600)
for i in range(spikes_c4.shape[0]):
    for j in spikes_c4[i]:
        if j == 1:
            sum_c4[i] += 1            


# plt.subplot(2,1,2)
# plt.plot(t, sum_c1)
# plt.plot(t, sum_c2)
# plt.plot(t, sum_c3)
# plt.plot(t, sum_c4)
# plt.plot(t, sum_c1 + sum_c2 + sum_c3 + sum_c4)
# plt.plot(t2, summed_spikes)

# res = minimize(swarm_fun, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
# c_res = minimize(constant, c_fopt, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})


# plt.plot(c_res.x)
# x0 = [res.x[0], res.x[1], res.x[2], res.x[3]]

# ll = -swarm_fun(x0)

# print(ll)
# print(c_res.x)

plt.show()