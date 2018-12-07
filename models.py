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

    def __init__(self, data):
        self.time_info = data['time_info']
        self.total_bins = round(
            (self.time_info.region_high - self.time_info.region_low) / (self.time_info.region_bin))

        #currently a hack to deal with time data given in seconds
        if self.time_info.converted:
            self.time_info.region_low /= 1000
            self.time_info.region_high /= 1000
            self.time_info.region_bin /= 1000
            self.time_info.converted = False

        self.t = np.arange(
            self.time_info.region_low,
            self.time_info.region_high,
            self.time_info.region_bin)

        # self.fit = None
        # self.fun = None
        # self.bounds = None
        # self.lb = None
        # self.ub = None

    def fit_params(self):
        """Fit model paramters using Particle Swarm Optimization then SciPy's minimize.

        Returns
        -------
        tuple of list and float
            Contains list of parameter fits and the final function value.

        """
        fit_pso, fun_pso = pso(
            self.build_function,
            self.lb, 
            self.ub,
            phip=0.5,
            phig=0.5,
            omega=0.5,
            minstep=1e-10,
            minfunc=1e-10,
            maxiter=800, #800 is arbitrary, doesn't seem to get reached
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

    def set_bounds(self, bounds):
        self.bounds = bounds
        self.lb = [x[0] for x in self.bounds]
        self.ub = [x[1] for x in self.bounds]


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

    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes_time']
        self.name = "time"
        self.num_params = 4
        self.region = self.t
        self.model_type = "time"
        # self.ut = None
        # self.st = None
        # self.a = None
        # self.o = None

        n = 2
        mean_delta = 0.10 * (self.time_info.region_high -
                             self.time_info.region_low)
        mean_bounds = (
            (self.time_info.region_low - mean_delta),
            (self.time_info.region_high + mean_delta))
        # mean_delta = 0.10 * (self.time_info.region_high -
        #     self.time_info.region_low)
        # mean_bounds = (
        #     (self.time_info.region_low - mean_delta),
        #     (self.time_info.region_high + mean_delta))
        # bounds = ((0.001, 1 / n), mean_bounds, (0.01, 5.0), (10**-10, 1 / n))
        #bounds = ((0.001, 1 / n), (-500, 2500), (0.01, 500), (10**-10, 1 / n))
        #bounds = ((0.001, 1 / n), mean_bounds, (0.01, 2000), (10**-10, 1 / n))



        # self.set_bounds(bounds)

    def build_function(self, x):
        #pso stores params in vector x
        a, ut, st, o = x[0], x[1], x[2], x[3]
 
        self.function = (
            (a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        res = np.sum(self.spikes * (-np.log(self.function)) +
                     (1 - self.spikes) * (-np.log(1 - (self.function))))
        return res

    def fit_params(self):
        super().fit_params()
        return (self.fit, self.fun)

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
        fun = (self.a * np.exp(-np.power(self.t - self.ut, 2.) /
                               (2 * np.power(self.st, 2.)))) + self.o
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

    def __init__(self, data):
        super().__init__(data)
        self.o = None
        self.spikes = data['spikes_time']
        self.model_type = "time"
        self.region = self.t

        self.name = "constant"
        self.num_params = 1
        bounds = ((10**-10, 0.99),)
        self.set_bounds(bounds)

    def build_function(self, x):
        o = x[0]
        return np.sum(self.spikes * (-np.log(o)) +
                      (1 - self.spikes) * (-np.log(1 - (o))))

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

    def __init__(
            self,
            spikes,
            time_info,
            bounds,
            time_params,
            conditions,
            pairs,
            num_trials):
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
        c2 = self.conditions[pair_2[0]] + self.conditions[pair_2[1]]

        big_t = (a1 * c1 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + \
            (a2 * c2 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))))

        result = np.sum(self.spikes * (-np.log(o + big_t.T)) +
                        (1 - self.spikes) * (-np.log(1 - (o + big_t.T))))
        return result

    def fit_params(self):
        super().fit_params()
        self.o = self.fit[0]
        self.a1 = self.fit[1]
        self.a2 = self.fit[2]

        return self.fit, self.fun

    def plot_fit(self, fit):
        ut, st, o = self.ut, self.st, self.o
        a1, a2 = self.a1, self.a2
        t = np.linspace(
            self.time_info.region_low,
            self.time_info.region_high,
            self.total_bins)
        fun = (a1 * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.))) + (
            a2 * np.exp(-np.power(t - ut, 2) / (2 * np.power(st, 2.))))) + o

        plt.plot(t, fun)


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

    def __init__(self, data):
        super().__init__(data)
        self.name = "category_time"
        self.region = self.t
        self.model_type = "time"
        #t ends up needing to include trial dimension due to condition setup
        self.t = np.tile(self.t, (data["num_trials"], 1))
        self.conditions = data["conditions"]
        self.spikes = data['spikes_time']
        self.ut = None
        self.st = None
        self.o = None
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.a4 = None
        self.bounds = "manually define!"
        self.num_params = 7
        n = 5
        mean_delta = 0.10 * (self.time_info.region_high -
                             self.time_info.region_low)
        mean_bounds = (
            (self.time_info.region_low - mean_delta),
            (self.time_info.region_high + mean_delta))
        bounds = (mean_bounds, (0.01, 5.0), (10**-10, 1 / n), (0.001, 1 / n), (0.001, 1 / n), (0.001, 1 / n), (0.001, 1 / n),)        
        self.set_bounds(bounds)


    def build_function(self, x):
        c1 = self.conditions[1]
        c2 = self.conditions[2]
        c3 = self.conditions[3]
        c4 = self.conditions[4]

        ut, st, o = x[0], x[1], x[2]
        a1, a2, a3, a4 = x[3], x[4], x[5], x[6]

        big_t = (a1 * c1 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
            a2 * c2 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
                a3 * c3 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.)))) + (
                    a4 * c4 * np.exp(-np.power(self.t.T - ut, 2.) / (2 * np.power(st, 2.))))

        return np.sum(self.spikes * (-np.log(o + big_t.T)) +
                      (1 - self.spikes) * (-np.log(1 - (o + big_t.T))))

    def fit_params(self):
        super().fit_params()
        return self.fit, self.fun

    def update_params(self):

        self.ut = self.fit[0]
        self.st = self.fit[1]
        self.o = self.fit[2]
        self.a1 = self.fit[3]
        self.a2 = self.fit[4]
        self.a3 = self.fit[5]
        self.a4 = self.fit[6]

    def pso_con(self, x):

        return 1 - (x[3] + x[4] + x[2] + x[5] + x[6])

    def expose_fit(self, category=0):
        ut, st, o = self.fit[0], self.fit[1], self.fit[2]
        cat_coefs = [self.fit[3], self.fit[4], self.fit[5], self.fit[6]]
        t = np.linspace(
            self.time_info.region_low,
            self.time_info.region_high,
            self.time_info.total_bins
        )
        if category:
            fun = (cat_coefs[category-1] * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.)))) 
        else:
        #dont use this
            fun = (cat_coefs[0] * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.)))) + (
                cat_coefs[1] * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.)))) + (
                    cat_coefs[2] * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.)))) + (
                        cat_coefs[3] * np.exp(-np.power(t - ut, 2.) / (2 * np.power(st, 2.)))) + o

        return fun


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

    def __init__(
            self,
            spikes,
            time_low,
            time_high,
            time_bin,
            bounds,
            conditions):
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
        return np.sum(self.spikes.T * (-np.log(big_t)) +
                      (1 - self.spikes.T) * (-np.log(1 - (big_t))))

    def fit_params(self):
        super().fit_params()
        self.a1 = self.fit[0]
        self.a2 = self.fit[1]
        self.a3 = self.fit[2]
        self.a4 = self.fit[3]

        return self.fit, self.fun

    def pso_con(self, x):
        return 1


class PositionTime(Model):

    def __init__(self, data):
        super().__init__(data)
        self.name = "time_position"
        #self.num_params = 8
        self.num_params = 10
        self.ut = None
        self.st = None
        self.a = None
        self.o = None
        n = 4
        mean_delta = 0.10 * (self.time_info.time_high - self.time_info.time_low)
        mean_bounds = (
            (self.time_info.time_low - mean_delta),
            (self.time_info.time_high + mean_delta))
        bounds = ((0.001, 1 / n), mean_bounds, (0.01, 5.0), (10**-10, 1 / n),
                    (0.001, 1 / n), mean_bounds, (0.01, 5.0), (10**-10, 1 / n),
                    mean_bounds, (0.01, 5.0)) 

        self.set_bounds(bounds)
        self.position = data["position"]

    def build_function(self, x):
        a_t, mu_t, sig_t, a_t0 = x[0], x[1], x[2], x[3]
        a_s, mu_s, sig_s, a_s0 = x[4], x[5], x[6], x[7]
        mu_sy, sig_sy= x[8], x[9]
        time_comp = (
            (a_t * np.exp(-np.power(self.t - mu_t, 2.) / (2 * np.power(sig_t, 2.)))) + a_t0)
        
        #spacial_comp = a_s * (np.exp(-np.power(self.position[0] - mu_s, 2.) / (2 * np.power(sig_s, 2.)))) + a_s0

        spacial_comp = a_s * (np.exp(-np.power(self.position[1] - mu_s, 2.) / (2 * np.power(sig_s, 2.))
            + (np.power(self.position[0] - mu_sy, 2.) / (2 * np.power(sig_sy, 2.))))) + a_s0


        self.function = time_comp + spacial_comp
        res = np.sum(self.spikes * (-np.log(self.function)) + 
            (1 - self.spikes) * (-np.log(1 - (self.function))))
        return res

    def fit_params(self):
        super().fit_params()
        return (self.fit, self.fun)

    def pso_con(self, x):
        return 1 - (x[0] + x[3] + x[4] + x[7])

    def expose_fit(self):
        if self.fit is None:
            raise ValueError("fit not yet computed")
        else:
            self.a_t = self.fit[0]
            self.mu_t = self.fit[1]
            self.sig_t = self.fit[2]
            self.a_t0 = self.fit[3]
            self.a_s = self.fit[4]
            self.mu_s = self.fit[5]
            self.sig_s = self.fit[6]
            self.a_s0 = self.fit[7]
            self.mu_sy = self.fit[8]
            self.sig_sy = self.fit[9]
        time_comp = (
            (self.a_t * np.exp(-np.power(self.t - self.mu_t, 2.) / (2 * np.power(self.sig_t, 2.)))) + self.a_t0)
        
        #spacial_comp = self.a_s * (np.exp(-np.power(self.position[0] - self.mu_s, 2.) / (2 * np.power(self.sig_s, 2.)))) + self.a_s0

        spacial_comp = self.a_s * (np.exp(-np.power(self.position[1] - self.mu_s, 2.) / (2 * np.power(self.sig_s, 2.))
            + (np.power(self.position[0] - self.mu_sy, 2.) / (2 * np.power(self.sig_sy, 2.))))) + self.a_s0
        fun = time_comp + spacial_comp
        return fun

class PositionGauss(Model):

    def __init__(self, data):
        super().__init__(data)
        self.name = "pos-gauss"
        #self.num_params = 8
        self.num_params = 4
        self.ut = None
        self.st = None
        self.a = None
        self.o = None
        n = 3
        mean_delta = 0.10 * (self.time_info.time_high - self.time_info.time_low)
        mean_bounds = (
            (self.time_info.time_low - mean_delta),
            (self.time_info.time_high + mean_delta))
        bounds = ((10**-10, 1 / n), (0, 1000), (0.01, 500.0), (10**-10,  1 / n))


        self.set_bounds(bounds)
        self.position = data["position"]

    def build_function(self, x):
        a, mu_x, sig_x, a_0 = x[0], x[1], x[2], x[3]
        xpos = np.arange(0, 995, 1)
        self.function = (a * (np.exp(-np.power(xpos - mu_x, 2.) / (2 * np.power(sig_x, 2.))))) + a_0 
        res = np.sum(self.spikes.T * (-np.log(self.function)) + 
            (1 - self.spikes.T) * (-np.log(1 - (self.function))))
        return res

    def fit_params(self):
        super().fit_params()
        return (self.fit, self.fun)

    def pso_con(self, x):
        return 1 - (x[0] + x[3])

    def expose_fit(self):
        if self.fit is None:
            raise ValueError("fit not yet computed")
        else:
            self.a = self.fit[0]
            self.mu_x = self.fit[1]
            self.sig_x = self.fit[2]
            self.a_0 = self.fit[3]


        xpos = np.arange(0, 990, 1)

        fun = self.a * (np.exp(-np.power(xpos - self.mu_x, 2.) / (2 * np.power(self.sig_x, 2.)))) + self.a_0
        return fun  

class BoxCategories(Model):
    def __init__(self, data):
        super().__init__(data)
        self.name = ("moving_box")
        n = 7
        self.model_type = "position"
        self.spikes = data['spikes_pos']
        self.pos_info = data['pos_info']
        self.num_params = 19
        self.conditions =  data["conditions"]
        self.x_pos = np.arange(0, 995, 1)
        self.x_pos =  np.tile(self.x_pos, (60, 1)).T
        # bounds = ((10**-10, 1 / n), (0, 1000), (1, 500.0), (10**-10, 1 / n), (0, 1000), (1, 500.0),
        #     (10**-10, 1 / n), (0, 1000), (1, 500.0), (10**-10, 1 / n), (0, 1000), (1, 500.0),
        #     (10**-10, 1 / n), (0, 1000), (1, 500.0), (10**-10, 1 / n), (0, 1000), (1, 500.0), 
        #     (10**-10,  1 / n))

        # self.set_bounds(bounds)
        self.region =  self.x_pos


    def build_function(self, x):
        a1, mu_x1, sig_x1 = x[0], x[1], x[2]
        a2, mu_x2, sig_x2 = x[3], x[4], x[5]
        a3, mu_x3, sig_x3 = x[6], x[7], x[8]
        a4, mu_x4, sig_x4 = x[9], x[10], x[11]
        a5, mu_x5, sig_x5 = x[12], x[13], x[14]
        a6, mu_x6, sig_x6 = x[15], x[16], x[17]
        a_0 = x[18]
        c1 = self.conditions[1]
        c2 = self.conditions[2]
        c3 = self.conditions[3]
        c4 = self.conditions[4]
        c5 = self.conditions[5]
        c6 = self.conditions[6]

        # xpos = np.arange(0, 995, 1)
        # xpos =  np.tile(xpos, (60, 1)).T



        pos1 = (a1 * c1 * np.exp(-np.power(self.x_pos - mu_x1, 2.) / (2 * np.power(sig_x1, 2.))))
        pos2 = (a2 * c2 * np.exp(-np.power(self.x_pos - mu_x2, 2.) / (2 * np.power(sig_x2, 2.))))
        pos3 = (a3 * c3 * np.exp(-np.power(self.x_pos - mu_x3, 2.) / (2 * np.power(sig_x3, 2.))))
        pos4 = (a4 * c4 * np.exp(-np.power(self.x_pos - mu_x4, 2.) / (2 * np.power(sig_x4, 2.))))
        pos5 = (a5 * c5 * np.exp(-np.power(self.x_pos - mu_x5, 2.) / (2 * np.power(sig_x5, 2.))))
        pos6 = (a6 * c6 * np.exp(-np.power(self.x_pos - mu_x6, 2.) / (2 * np.power(sig_x6, 2.))))

        self.function = pos1 + pos2 + pos3 + pos4 + pos5 + pos6 + a_0 
        res = np.sum(self.spikes * (-np.log(self.function.T)) + 
            (1 - self.spikes) * (-np.log(1 - (self.function.T))))
        return res

    def update_params(self):
        self.ut = self.fit[0]
        self.st = self.fit[1]
        self.a = self.fit[2]
        self.o = self.fit[3]

    def fit_params(self):
        super().fit_params()
        return (self.fit, self.fun)

    def pso_con(self, x):
        return 1 - (x[0] + x[3])

    def expose_fit(self, category=0):

        if self.fit is None:
            raise ValueError("fit not yet computed")
        else:
            a1, mu_x1, sig_x1 = self.fit[0], self.fit[1], self.fit[2]
            a2, mu_x2, sig_x2 = self.fit[3], self.fit[4], self.fit[5]
            a3, mu_x3, sig_x3 = self.fit[6], self.fit[7], self.fit[8]
            a4, mu_x4, sig_x4 = self.fit[9], self.fit[10], self.fit[11]
            a5, mu_x5, sig_x5 = self.fit[12], self.fit[13], self.fit[14]
            a6, mu_x6, sig_x6 = self.fit[15], self.fit[16], self.fit[17]
            a_0 = self.fit[18]

            cat_coefs = [a1, a2, a3, a4, a5, a6]
            mu_cat = [mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, mu_x6]
            sig_cat = [sig_x1, sig_x2, sig_x3, sig_x4, sig_x5, sig_x6]

            fun = cat_coefs[category-1] * np.exp(-np.power(self.x_pos - mu_cat[category-1], 2.) / (2 * np.power(sig_cat[category-1], 2.)))

            return fun