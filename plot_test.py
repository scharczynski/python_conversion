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
import time_cells
import cat_time_cells 

a1 = 0.0025
a2 = 0.005
a3 = 0.0047
a4 = 0.001
ut = 0.626
st = 0.1
o = 0.00452

# [-0.55568783  0.10000102  0.00492764  0.04065333  0.06859002  0.15504045
#   0.07720791]

# 0.021205502853744812 0.06699635359413272 0.04341807002445048 0.07503794265149492
# -0.5833666407010306 0.1 0.004927745413629169

# [0.62574579 0.1        0.00451964 0.00249534 0.00521065 0.00476935
#  0.001     ]

#t = np.arange(1600)
t = np.arange(0,1.6,0.001)
#print(t)
plt.plot(t,(o + (a1 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
    a2 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))) + (
            a3 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
                a4 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.)))))))))

#plt.plot(o + (a1 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.)))))

plt.show()