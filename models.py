"""Module containing all model classes.

    These models are currently based off those used in Compressed timeline of recent experience in monkey lPFC 
    (Tiganj et al. 2018)

    As a consequence, models currently will be largely coupled to the specific reference experiement.

    Models are constructed to deal with one cell at a time.


"""

import numpy as np
from pyswarm import pso
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from math import isnan
import pyswarms as ps


class Model(object):

    """Base class for models.

    Provides common class attributes.

    Parameters
    ----------
    spikes : numpy.ndarray
        Array of binary spike train data, of dimension (trials × time).
    time_low : float 
        Time, in seconds, beginning the window of interest.
    time_high : float
        Time, in seconds, ending the window of interest.
    time_bin : float
        Time, in seconds, which describes the size of bins spike data is collected into.
    x0 : list
        List of initial guesses for the L-BFGS-B optimization algorithm
    bounds : tuple 
        Tuple of length number of paramters for the given model, setting upper and lower 
        bounds on parameters

    Attributes
    ----------
    spikes : numpy.ndarray
        Array of binary spike train data, of dimension (trials × time).
    bounds : tuple 
        Tuple of length number of paramters for the given model, setting upper and lower 
        bounds on parameters.
    t : numpy.ndarray
        Array of timeslices of size specified by time_low, time_high and time_bin.
    fit : list
        List of parameter fits after fitting process has been completed, initially None.
    fun : float
        Value of model objective function at computed fit parameters.
    lb : list
        List of parameter lower bounds.
    ub : list
        List of parameter upper bounds.
        
    """

    def __init__(self, cell_no, spikes, time_info, bounds):
        self.cell_no = cell_no
        self.spikes = spikes   
        self.bounds = bounds
        total_bins  = ((time_info.time_high - time_info.time_low) /  (time_info.time_bin))
        self.t = np.linspace(time_info.time_low, time_info.time_high, total_bins)
        self.fit = None
        self.fun = None
        self.lb = [x[0] for x in bounds]
        self.ub = [x[1] for x in bounds]
    def fit_params(self):
        """Fit model paramters using Particle Swarm Optimization then SciPy's minimize.

        Returns
        -------
        tuple of list and float
            Contains list of parameter fits and the final function value.

        """
        fit_pso, fun_pso = pso(
            self.build_function, 
            self.lb, self.ub, 
            maxiter=800, 
            f_ieqcons=self.pso_con
            )
        second_pass_res = minimize(
            self.build_function, 
            fit_pso, 
            method='L-BFGS-B',
            bounds=self.bounds, 
            options={'disp': False}
            )
        # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=4, options=options)
        # best_cost, best_pos = optimizer.optimize(self.pyswarms_funct, iters=100, verbose=3, print_step=25, t=self.t)
        # self.fit = best_pos
        # self.fun = best_cost
        # x0 = np.asarray([0.1,0.1,0.1, 0.1, ])
        # second_pass_res = minimize(self.build_function, self.x0, method='L-BFGS-B', bounds=self.bounds, options={'disp': False})
        self.fit = second_pass_res.x
        self.fun = second_pass_res.fun
        return (self.fit, self.fun)

    def build_function(self):
        """Embed model parameters in model function.

        Parameters
        ----------
        x : numpy.ndarray
            Contains optimization parameters at intermediate steps.
        
        Returns
        -------
        float
            Negative log-likelihood of given model under current parameter choices.

        """
        raise NotImplementedError("Must override build_function")

    def pso_con(self):
        """Define constraint on coefficients for PSO

        Note
        ----
        Constraints for pyswarm module take the form of an array of equations 
        that sum to zero.    
    
        Parameters
        ----------
        x : numpy.ndarray
            Contains optimization parameters at intermediate steps.

        """
        raise NotImplementedError("Must override pso_con")

    def plot_fit(self):
        """Plot model fit.

        """
        raise NotImplementedError("Must override plot_fit")


class Time(Model):

    """Model which contains a time dependent gaussian compenent and an offset parameter.

    Parameters
    ----------
    spikes : numpy.ndarray
        Array of binary spike train data, of dimension (trials × time).
    time_low : float 
        Time, in seconds, beginning the window of interest.
    time_high : float
        Time, in seconds, ending the window of interest.
    time_bin : float
        Time, in seconds, which describes the size of bins spike data is collected into.
    x0 : list
        List of initial guesses for the L-BFGS-B optimization algorithm
    bounds : tuple 
        Tuple of length number of paramters for the given model, setting upper and lower 
        bounds on parameters

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    ut : float
        Mean of gaussian distribution.
    st : float
        Standard deviation of gaussian distribution.
    a : float
        Coefficient of guassian distribution.
    o : float
        Additive offset of distribution.

    """

    def __init__(self, cell_no, spikes, time_info, bounds):
        super().__init__(cell_no, spikes, time_info, bounds)
        self.name = "Time"
        self.ut = None
        self.st = None
        self.a = None
        self.o = None
        self.x0 = [0.1, 0.1, 0.1, 0.01]

    # def compute_funct(self, x):
    #     a, ut, st, o = x[0], x[1], x[2], x[3]
    #     T = a*np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))
    #     res = np.sum(self.spikes*(-np.log(o+T))+(1-self.spikes)*(-np.log(1-(o+T))))
    #     if isnan(res):
    #         print("is nan     ")
    #         print(T)
    #         print(a, ut, st, o)

    #     # return np.sum(self.spikes*(-np.log(o+T))+(1-self.spikes)*(-np.log(1-(o+T))))
    #     return res

    def pyswarms_funct(self, x, t):
        a, ut, st, o = x[:,0], x[:,1], x[:,2], x[:,3]
        self.function = ((a*np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        res = np.sum(self.spikes*(-np.log(self.function)) + (1-self.spikes)*(-np.log(1-(self.function))))
        return res

    def build_function(self, x):
        a, ut, st, o = x[0], x[1], x[2], x[3]
        self.function = ((a*np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        # print(a, ut, st, o)
        # print(a*np.exp(-np.power(0 - ut, 2.) / ((2 * np.power(st, 2.))) + o))
        # if any(np.where(self.function==0)[0]):
        #     print ("Zero")
        #     print (np.where(self.function==0)[0].size)
        #     print(np.where(self.function==0)[0])
        #     print(a, ut, st, o)
        # if any(np.where(self.function==1)[0]):
        #     print ("One")
        #     print (np.where(self.function==1)[0].size)
        #     print (np.where(self.function==1)[0])
        #     print(a, ut, st, o)

        # if any(np.where(self.function < 0)[0]):
        #     print ("Negative")
        #     print (np.where(self.function<0)[0].size)
        #     print (np.where(self.function<0)[0])
        #     print(a,ut,st,o)
        res = np.sum(self.spikes*(-np.log(self.function)) + (1-self.spikes)*(-np.log(1-(self.function))))
        # return np.sum(self.spikes*(-np.log(self.function))+(1-self.spikes)*(-np.log(1-(self.function))))
        return res

    def fit_params(self):
        super().fit_params()
        self.a = self.fit[0]
        self.ut = self.fit[1]
        self.st = self.fit[2]
        self.o = self.fit[3]
        return (self.fit, self.fun)

    # def build_objective(self, x):
    #     return 
    # def fit_params(self):

    #     fits = minimize(self.compute_funct, self.x0, method='L-BFGS-B', bounds=self.bounds, options={'xtol': 1e-8, 'disp': True})

    #     # self.fit = fits
    #     # self.fun = function_value
    #     self.fit = fits.x
    #     self.fun = fits.fun

    #     return fits.x, fits.fun
    #     # return fits, function_value

    # def fit_params_pso(self):
    #     fits, function_value = pso(self.compute_funct, self.lb, self.ub, maxiter=600)
    #     # fits, function_value = pso(self.compute_funct, self.lb, self.ub, maxiter=600, f_ieqcons=self.pso_con)

    #     fits = minimize(self.compute_funct, fits, method='L-BFGS-B', bounds=self.bounds, options={'disp': False})

    #     fits = minimize(self.compute_funct, self.x0, method='L-BFGS-B', bounds=self.bounds, options={'disp': False})
    #     fits_pso, fun_pso = pso(self.compute_funct, self.lb, self.ub, maxiter=600, f_ieqcons=self.pso_con)
    #     if fun_pso < fits.fun:
    #         self.fit = fits_pso
    #         self.fun = fun_pso
    #     else:
    #         self.fit = fits.x
    #         self.fun = fits.fun

    #     return fits.x, fits.fun

    def pso_con(self, x):
        return 1 - (x[0] + x[3])

    def plot_fit(self):
        if self.fit is None:
            raise ValueError("fit not yet computed")

        #a, ut, st, o = self.fit[0], self.fit[1], self.fit[2], self.fit[3]
        fun = (self.a*np.exp(-np.power(self.t - self.ut, 2.) / (2 * np.power(self.st, 2.))) + self.o)
        plt.plot(self.t, fun)



class Const(Model):

    """Model which contains only a single offset parameter.

    Parameters
    ----------
    spikes : numpy.ndarray
        Array of binary spike train data, of dimension (trials × time).
    time_low : float 
        Time, in seconds, beginning the window of interest.
    time_high : float
        Time, in seconds, ending the window of interest.
    time_bin : float
        Time, in seconds, which describes the size of bins spike data is collected into.
    x0 : list
        List of initial guesses for the L-BFGS-B optimization algorithm
    bounds : tuple 
        Tuple of length number of paramters for the given model, setting upper and lower 
        bounds on parameters

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    o : float
        Additive offset of distribution.

    """

    def __init__(self, cell_no, spikes, time_info, bounds):
        super().__init__(cell_no, spikes, time_info, bounds)
        self.o = None
        self.name  = "Constant"
        self.x0 = [0.001]

    def build_function(self, x):
        o = x[0]
        return np.sum(self.spikes*(-np.log(o)) + (1-self.spikes)*(-np.log(1-(o))))

    def fit_params(self):
        super().fit_params()
        self.o = self.fit
        return (self.fit, self.fun)
    
    def pso_con(self, x):
        return 1 - x
        
    def plot_fit(self):
        plt.axhline(y=self.fit, color='r', linestyle='-')

class CatSetTime(Model):

    """Model which contains seperate time-dependent gaussian terms per each given category sets.

    Parameters
    ----------
    spikes : numpy.ndarray
        Array of binary spike train data, of dimension (trials × time).
    time_low : float 
        Time, in seconds, beginning the window of interest.
    time_high : float
        Time, in seconds, ending the window of interest.
    time_bin : float
        Time, in seconds, which describes the size of bins spike data is collected into.
    x0 : list
        List of initial guesses for the L-BFGS-B optimization algorithm
    bounds : tuple 
        Tuple of length number of paramters for the given model, setting upper and lower 
        bounds on parameters
    time_params : list
        List of gaussian parameters from a previous time-only fit.
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.

    Attributes
    ----------
    t : numpy.ndarray
        Array of timeslices of size specified by time_low, time_high and time_bin.
        This array is repeated a number of times equal to the amount of trials 
        this cell has.
    name : string
        Human readable string describing the model.
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.
    ut : float
        Mean of gaussian distribution.
    st : float
        Standard deviation of gaussian distribution.
    a1 : float
        Coefficient of category set 1 gaussian distribution.
    a2 : float
        Coefficient of category set 2 gaussian distribution.
    o : float
        Additive offset of distribution.

    """

    def __init__(self, cell_no, spikes, time_low, time_high, time_bin, bounds, time_params, conditions, pairs):
        super().__init__(cell_no, spikes, time_low, time_high, time_bin, bounds)
        #REWRITE TO INCLUDE CELL NO
        self.pairs = pairs
        self.t = np.tile(self.t, (1064, 1)) 
        self.conditions = conditions
        self.ut = time_params[1]
        self.st = time_params[2]
        self.a1 = None
        self.a2 = None
        self.o = None

    def build_function(self, x):
        ut, st, o = self.ut, self.st, x[0]
        a1, a2 = x[1], x[2]
        pair_1 = self.pairs[0]
        pair_2 = self.pairs[1]
        c1 = self.conditions[pair_1[0], self.cell_no] + self.conditions[pair_1[1], self.cell_no]
        c2 = self.conditions[pair_2[0], self.cell_no] + self.conditions[pair_2[1], self.cell_no]

        big_t = (a1 * c1 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
            a2 * c2 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))))
                

        result = np.sum(self.spikes*(-np.log(o+big_t.T))+(1-self.spikes)*(-np.log(1-(o+big_t.T))))
        return result

    def fit_params(self):

        x0 = [0.1,0.1,0.1]

        bound = ((10**-10, 0.2), (0, 0.2), (0, 0.2))

        fits = minimize(self.compute_funct, x0, args=(pairs), method='L-BFGS-B', bounds=bound, options={'disp': True})


        return fits.x, fits.fun

    def plot_fit(self, fit):
        ut, st, o = self.ut, self.st, fit[0]
        a1, a2 =  fit[1], fit[2]
        t = np.linspace(0, 1.6, 1600)
        fun = (a1 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
            a2 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))))) + o

        plt.plot(t, fun)

    def plot_raster(self, bin_spikes):
        ax = sns.heatmap(bin_spikes)


class CatTime(Model):

    """Model which contains seperate time-dependent gaussian terms per each given category.

    Parameters
    ----------
    spikes : numpy.ndarray
        Array of binary spike train data, of dimension (trials × time).
    time_low : float 
        Time, in seconds, beginning the window of interest.
    time_high : float
        Time, in seconds, ending the window of interest.
    time_bin : float
        Time, in seconds, which describes the size of bins spike data is collected into.
    x0 : list
        List of initial guesses for the L-BFGS-B optimization algorithm
    bounds : tuple 
        Tuple of length number of paramters for the given model, setting upper and lower 
        bounds on parameters
    time_params : list
        List of gaussian parameters from a previous time-only fit.
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.

    Attributes
    ----------
    t : numpy.ndarray
        Array of timeslices of size specified by time_low, time_high and time_bin.
        This array is repeated a number of times equal to the amount of trials 
        this cell has.
    name : string
        Human readable string describing the model.
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.
    ut : float
        Mean of gaussian distribution.
    st : float
        Standard deviation of gaussian distribution.
    a1 : float
        Coefficient of category 1 gaussian distribution.
    a2 : float
        Coefficient of category 2 gaussian distribution.
    a3 : float
        Coefficient of category 3 gaussian distribution.
    a4 : float
        Coefficient of category 4 gaussian distribution.
    o : float
        Additive offset of distribution.

    """

    def __init__(self, cell_no, spikes, time_info, bounds, time_params, conditions):
        super().__init__(cell_no, spikes, time_info, bounds)
        self.name = "Category-Time"
        #REWRITE TO CONSIDER TRIALS PER CELL
        self.t = np.tile(self.t, (1064, 1))
        self.conditions = conditions
        self.ut = time_params[1]
        self.st = time_params[2]
        self.o = None
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.a4 = None
        self.bounds = bounds

    def build_function(self, x):
        #REWRITE TO CONSIDER CELL_NO
        c1 = self.conditions[1, self.cell_no]
        c2 = self.conditions[2, self.cell_no]
        c3 = self.conditions[3, self.cell_no]
        c4 = self.conditions[4, self.cell_no]
        ut, st, o = self.ut, self.st, x[0]
        a1, a2, a3, a4 = x[1], x[2], x[3], x[4]

        big_t = a1 * c1 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))) + (
            a2 * c2 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))) + (
                a3 * c3 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))) + (
                    a4 * c4 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))))))

        return np.sum(self.spikes*(-np.log(o+big_t.T))+(1-self.spikes)*(-np.log(1-(o+big_t.T))))

    def fit_params(self):
        super().fit_params()
        self.o = self.fit[0]
        self.a1 = self.fit[1]
        self.a2 = self.fit[2]
        self.a3 = self.fit[3]
        self.a4 = self.fit[4]

        return self.fit, self.fun

    
    def fit_params_pso(self):
        fits, function_value = pso(self.compute_funct, self.lb, self.ub, maxiter=100, f_ieqcons=self.pso_con)
        fits = minimize(self.compute_funct, fits, method='L-BFGS-B', bounds=self.bounds, options={'xtol': 1e-8, 'disp': True})
        self.fit = fits.x
        self.fun = fits.fun
        return fits.x, fits.fun
        
    def pso_con(self, x):
 
        return 1 - (x[0] + x[1] + x[2] + x[3] + x[4])

    def plot_fit(self):
        ut, st, o = self.ut, self.st, self.fit[0]
        a1, a2, a3, a4 = self.fit[1], self.fit[2], self.fit[3], self.fit[4]
        #plt.subplot(2,1,1)
        t = np.linspace(0, 1.6, 1600)
        fun = (a1 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
            a2 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))) + (
                a3 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
                    a4 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))))))) + o


        plt.plot(t, fun)

    def plot_cat_fit(self, fit, cat):
        ut, st, o = self.ut, self.st, fit[0]
        cat_coeff = fit[cat]

        t = np.linspace(0, 1.6, 1600)

        fun = (cat_coeff * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.)))) + o

        plt.plot(t, fun) 

    def plot_c1_fit(self, fit):
        self.plot_cat_fit(fit, 1)

    def plot_c2_fit(self, fit):
        self.plot_cat_fit(fit, 2)

    def plot_c3_fit(self, fit):
        self.plot_cat_fit(fit, 3)

    def plot_c4_fit(self, fit):
        self.plot_cat_fit(fit, 4)

    def plot_all(self, fit):
        
        plt.subplot(2, 2, 1)
        self.plot_c1_fit(fit)

        plt.subplot(2,2,2)
        self.plot_c2_fit(fit)

        plt.subplot(2,2,3)
        self.plot_c3_fit(fit)

        plt.subplot(2, 2, 4)
        self.plot_c4_fit(fit)

    def plot_raster(self, bin_spikes):
        ax = sns.heatmap(bin_spikes)

    def plot_raster_c1(self, bin_spikes):
        ax1 = sns.heatmap(bin_spikes.T * self.c1)

    def plot_raster_c2(self, bin_spikes):
        ax2 = sns.heatmap(bin_spikes.T * self.c2)

    def plot_raster_c3(self, bin_spikes):
        ax3 = sns.heatmap(bin_spikes.T * self.c3)

    def plot_raster_c4(self, bin_spikes):
        ax4 = sns.heatmap(bin_spikes.T * self.c4)


class ConstCat(Model):

    """Model which contains seperate constant terms per each given category.

    Parameters
    ----------
    spikes : numpy.ndarray
        Array of binary spike train data, of dimension (trials × time).
    time_low : float 
        Time, in seconds, beginning the window of interest.
    time_high : float
        Time, in seconds, ending the window of interest.
    time_bin : float
        Time, in seconds, which describes the size of bins spike data is collected into.
    x0 : list
        List of initial guesses for the L-BFGS-B optimization algorithm
    bounds : tuple 
        Tuple of length number of paramters for the given model, setting upper and lower 
        bounds on parameters
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    a1 : float
        Coefficient of category 1 gaussian distribution.
    a2 : float
        Coefficient of category 2 gaussian distribution.
    a3 : float
        Coefficient of category 3 gaussian distribution.
    a4 : float
        Coefficient of category 4 gaussian distribution.

    """

    def __init__(self, cell_no, spikes, time_low, time_high, time_bin, bounds, conditions):
        super().__init__(cell_no, spikes, time_low, time_high, time_bin, bounds)
        self.name = "Constant-Category"
        self.conditions = conditions
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.a4 = None

    def build_function(self, x):
        c1 = self.conditions[1, self.cell_no]
        c2 = self.conditions[2, self.cell_no]
        c3 = self.conditions[3, self.cell_no]
        c4 = self.conditions[4, self.cell_no]
        a1, a2, a3, a4 = x[0], x[1], x[2], x[3]
        big_t = a1 * c1 + a2 * c2 + a3 * c3 + a4 * c4
        return np.sum(self.spikes.T*(-np.log(big_t))+(1-self.spikes.T)*(-np.log(1-(big_t))))

    def fit_params(self):
        super().fit_params()
        self.a1 = self.fit[0]
        self.a2 = self.fit[1]
        self.a3 = self.fit[2]
        self.a4 = self.fit[3]

        return self.fit, self.fun
    
    def pso_con(self, x):
        return 1

