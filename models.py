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
    time_info : TimeInfo 
        Object that holds timing information including the beginning and end of the region
        of interest and the time bin. All in seconds.
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
    time_info : TimeInfo 
        Object that holds timing information including the beginning and end of the region
        of interest and the time bin. All in seconds.
    total_bins : int
        Calculated value determining total time bins analyzed.
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

    def __init__(self, spikes, time_info, bounds):
        self.spikes = spikes   
        self.bounds = bounds
        self.time_info = time_info
        self.total_bins  = ((time_info.time_high - time_info.time_low) /  (time_info.time_bin))
        self.t = np.linspace(time_info.time_low, time_info.time_high, self.total_bins)
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

    def expose_fit(self):
        """Plot model fit.

        """
        raise NotImplementedError("Must override plot_fit")


class Time(Model):

    """Model which contains a time dependent gaussian compenent and an offset parameter.

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    num_params : int
        Integer signifying the number of model parameters.
    ut : float
        Mean of gaussian distribution.
    st : float
        Standard deviation of gaussian distribution.
    a : float
        Coefficient of guassian distribution.
    o : float
        Additive offset of distribution.

    """

    def __init__(self, spikes, time_info, bounds):
        super().__init__(spikes, time_info, bounds)
        self.name = "time"
        self.num_params = 4
        self.ut = None
        self.st = None
        self.a = None
        self.o = None

    def build_function(self, x):
        a, ut, st, o = x[0], x[1], x[2], x[3]
        self.function = ((a*np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        res = np.sum(self.spikes*(-np.log(self.function)) + (1-self.spikes)*(-np.log(1-(self.function))))
        return res

    def fit_params(self):
        super().fit_params()
        return (self.fit, self.fun)

    def update_params(self):
        self.ut = self.fit[0]
        self.st = self.fit[1]
        self.a = self.fit[2]
        self.o = self.fit[3]


    def pso_con(self, x):
        return 1 - (x[0] + x[3])

    def expose_fit(self):
        if self.fit is None:
            raise ValueError("fit not yet computed")
        else:
            self.a = self.fit[0]
            self.ut = self.fit[1]
            self.st = self.fit[2]
            self.o = self.fit[3]
        fun = (self.a*np.exp(-np.power(self.t - self.ut, 2.) / (2 * np.power(self.st, 2.))) + self.o)
        return fun



class Const(Model):

    """Model which contains only a single offset parameter.

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    o : float
        Additive offset of distribution.
    num_params : int
        Integer signifying the number of model parameters.

    """

    def __init__(self, spikes, time_info, bounds):
        super().__init__(spikes, time_info, bounds)
        self.o = None
        self.name  = "constant"
        self.num_params = 1

    def build_function(self, x):
        o = x[0]
        return np.sum(self.spikes*(-np.log(o)) + (1-self.spikes)*(-np.log(1-(o))))

    def fit_params(self):
        super().fit_params()
        self.o = self.fit
        return (self.fit, self.fun)
    
    def pso_con(self, x):
        return 1 - x
        
    def expose_fit(self):
        if self.fit is None:
            raise ValueError("fit not yet computed")
        return self.fit
        
class CatSetTime(Model):

    """Model which contains seperate time-dependent gaussian terms per each given category sets.

    Parameters
    ----------
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

    def __init__(self, spikes, time_info, bounds, time_params, conditions, pairs, num_trials):
        super().__init__(spikes, time_info, bounds)
        self.pairs = pairs
        self.t = np.tile(self.t, (num_trials, 1)) 
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
        c1 = self.conditions[pair_1[0]] + self.conditions[pair_1[1]]
        c2 = self.conditions[pair_2[0]]+ self.conditions[pair_2[1]]

        big_t = (a1 * c1 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
            a2 * c2 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))))
                

        result = np.sum(self.spikes*(-np.log(o+big_t.T))+(1-self.spikes)*(-np.log(1-(o+big_t.T))))
        return result

    def fit_params(self):
        super().fit_params()
        self.o = self.fit[0]
        self.a1 = self.fit[1]
        self.a2 = self.fit[2]

        return self.fit, self.fun

    def plot_fit(self, fit):
        ut, st, o = self.ut, self.st, self.o
        a1, a2 =  self.a1, self.a2
        t = np.linspace(self.time_info.time_low, self.time_info.time_high, self.total_bins)
        fun = (a1 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
            a2 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))))) + o

        plt.plot(t, fun)

    def plot_raster(self, bin_spikes):
        ax = sns.heatmap(bin_spikes)


class CatTime(Model):

    """Model which contains seperate time-dependent gaussian terms per each given category.

    Parameters
    ----------
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

    def __init__(self, spikes, time_info, bounds, time_params, conditions, num_trials):
        super().__init__(spikes, time_info, bounds)
        self.name = "category_time"
        self.t = np.tile(self.t, (num_trials, 1))
        self.conditions = conditions
        self.ut = time_params[1]
        self.st = time_params[2]
        self.o = None
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.a4 = None
        self.bounds = bounds
        self.num_params = 7

    def build_function(self, x):
        c1 = self.conditions[1]
        c2 = self.conditions[2]
        c3 = self.conditions[3]
        c4 = self.conditions[4]

        ut, st, o = self.ut, self.st, x[0]
        a1, a2, a3, a4 = x[1], x[2], x[3], x[4]

        big_t = (a1 * c1 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
            a2 * c2 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
                a3 * c3 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
                    a4 * c4 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))))


        return np.sum(self.spikes*(-np.log(o+big_t.T))+(1-self.spikes)*(-np.log(1-(o+big_t.T))))

    def fit_params(self):
        super().fit_params()
        return self.fit, self.fun
    
    def update_params(self):
        self.o = self.fit[0]
        self.a1 = self.fit[1]
        self.a2 = self.fit[2]
        self.a3 = self.fit[3]
        self.a4 = self.fit[4]


    def pso_con(self, x):
 
        return 1 - (x[0] + x[1] + x[2] + x[3] + x[4])

    def plot_fit(self):
        ut, st, o = self.ut, self.st, self.o
        a1, a2, a3, a4 = self.a1, self.a2, self.a3, self.a4
        t = np.linspace(self.time_info.time_low, self.time_info.time_high, self.total_bins)

        fun = (a1 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
            a2 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))) + (
                a3 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
                    a4 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))))))) + o


        plt.plot(t, fun)

  

class ConstCat(Model):

    """Model which contains seperate constant terms per each given category.

    Parameters
    ----------
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    conditions : dict (int: numpy.ndarray of int)
        Dictionary containing trial conditions per trial per cell.
    a1 : float
        Coefficient of category 1 gaussian distribution.
    a2 : float
        Coefficient of category 2 gaussian distribution.
    a3 : float
        Coefficient of category 3 gaussian distribution.
    a4 : float
        Coefficient of category 4 gaussian distribution.

    """

    def __init__(self, spikes, time_low, time_high, time_bin, bounds, conditions):
        super().__init__(spikes, time_low, time_high, time_bin, bounds)
        self.name = "Constant-Category"
        self.conditions = conditions
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.a4 = None

    def build_function(self, x):
        c1 = self.conditions[1]
        c2 = self.conditions[2]
        c3 = self.conditions[3]
        c4 = self.conditions[4]
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

