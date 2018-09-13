import numpy as np
import parse_matlab_data
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Config(object):

    def __init__(self, plot, max_p, data, n, batch_job):

        self.plot = plot
        self.max_p = max_p
        self.data = data
        self.n = n
        self.batch_job = batch_job

        if (batch_job):
            self.plot = False

class Solver(object):

    def __init__(self, test_train_flag, config, function, options):

        self.test_train_flag = test_train_flag
        self.config = config
        self.function = function
        self.options = options

class Param(object):

    def __init__(self, name, lb, ub):

        self.name = name
        self.ub = ub
        self.lb = lb
        self.val = 0.1

class Params(object):

    def __init__(self, params):
        self.params = params
        self.bound_dict = {}
        for param in params:
            self.bound_dict[param.name] = param
        self.params = self.bound_dict
    
class Function(object):

    def __init__(self, function, log_likelyhood, params):
        self.params = params
        self.function = function
        self.log_likelyhood = log_likelyhood


class FTimeCells(object):

    def __init__(self, params):
        print (type(params))
        self.params = params.params
        self.ut = self.params['ut']
        self.st = self.params['st']
        self.a = self.params['a']
        self.t = np.arange(0,1600)
        self.offset = self.params['offset']

        # self.funct = self.a.val*np.exp((-(self.t - self.ut.val)**2)/(2*self.st.val**2))
        #self.funct = params[0]*np.exp((-(self.t - params[1])**2)/(2*params[2]**2))

    def build_function(self, x):

        funct = x[0]*np.exp((-(self.t - x[1])**2)/(2*x[2]**2))

        #funct = self.a*math.exp((-(self.t - self.ut)**2)/(2*self.st**2))

        return funct

    def log_likelyhood(self, x, f_spikes, params):
        
        #T = self.funct
        T = build_function(x)

        #return sum(f_spikes*(-np.log(self.offset.val+T))+(1-f_spikes)*(-np.log(1-(self.offset.val+T))))
        return sum(f_spikes*(-np.log(x[3]+T))+(1-f_spikes)*(-np.log(1-(x[3]+T))))

#x = FTimeCells()

spikes, conditions, num_trials, trial_length, summed_spikes  = parse_matlab_data.import_data()
config = Config(False, 0.05, summed_spikes, 1, False)


ut = Param("ut", 0.05, 10)
st = Param('st', 0.05, 10)
a = Param('a', 0.01, 10)
offset = Param('offset', 0.01, 10)

param_list = (ut, st, a, offset)

params = Params(param_list)
params2 = [0.11, 0.01, 0.01, 0.01]
cell = FTimeCells(params)

ll = cell.log_likelyhood(x, summed_spikes, params2)

res = minimize(ll, params2,  method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
# print (ll)
# plt.plot(ll)
# plt.show()

