import numpy as np
from pyswarm import pso
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


class ConstCatCell(object):

    def __init__(self, spikes, conditions):

        self.spikes = spikes
        self.conditions = conditions

        self.c1 = conditions[1][0]
        self.c2 = conditions[2][0]
        self.c3 = conditions[3][0]
        self.c4 = conditions[4][0]

    def compute_funct(self, x):

        a1, a2, a3, a4 = x[0], x[1], x[2], x[3]

        big_t = a1 * self.c1 + a2 * self.c2 + a3 * self.c3 + a4 * self.c4
        print (big_t)
        return np.sum(self.spikes.T*(-np.log(big_t))+(1-self.spikes.T)*(-np.log(1-(big_t))))

    def fit_params(self):

        x0 = [0.01, 0.01, 0.01, 0.01]
        bound = ((10**-10, 0.25), (10**-10, 0.25), (10**-10, 0.25), (10**-10, 0.25))

        fits = minimize(self.compute_funct, x0, method='L-BFGS-B', bounds=bound, options={'disp': True})

        return fits.x, fits.fun

