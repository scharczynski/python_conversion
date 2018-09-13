import numpy as np
from pyswarm import pso
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import cvxpy as cvx


class TimeCell(object):

    def __init__(self, spikes, time_range, time_bin):
        self.name = "Time"
        self.spikes = spikes
        self.t = np.linspace(0, 1.6, 1600)
        self.fits = None
        self.fun = None

    def compute_funct(self, x):
        a, ut, st, o = x[0], x[1], x[2], x[3]
        T = a*np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))

        return np.sum(self.spikes*(-np.log(o+T))+(1-self.spikes)*(-np.log(1-(o+T))))

    def fit_params(self):

        #fits, function_value = pso(self.compute_funct, self.lb, self.ub, maxiter=1400)
        x0 = [0.1, 0.5, 0.5, 0.01]
        bound = ( (0.001, 0.2), (0.1, 0.9), (0.1, 5.0), (10**-10, 5))
        fits = minimize(self.compute_funct, x0, method='L-BFGS-B', bounds=bound, options={'xtol': 1e-8, 'disp': True})
        print (fits.x)
        return fits.x, fits.fun

    
    def plot_fit(self, fit):
        a, ut, st, o = fit[0], fit[1], fit[2], fit[3]
        fun = (a*np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o
        plt.subplot(2,1,1)

        plt.plot(self.t, fun)
        

    # def tf_fit(self):


    #     ut = tf.Variable(0.2, trainable=True)
    #     st = tf.Variable(0.3, trainable=True)
    #     o = tf.Variable(0.001, trainable=True)
    #     a = tf.Variable(0.001, trainable=True)

    #     T = a*np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))

    #     loss = np.sum(self.spikes*(-np.log(o+T))+(1-self.spikes)*(-np.log(1-(o+T))))

    #     opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         for i in range(100):
    #             print(sess.run([(ut,st,o,a),loss]))
    #             sess.run(opt)