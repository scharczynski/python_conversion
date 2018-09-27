import numpy as np
import scipy.io as sio
import math
import time
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import filters
import scipy.optimize as op
import scipy.ndimage
from math import factorial



def import_data():
    path  = "/Users/stevecharczynski/workspace/Buffalo/data_cromer.mat"
    output = sio.loadmat(path)

    data = output['data']
    spikes = data['spikes'][0][0][:]
    conditions = data['conditions'][0][0][:]
    num_trials = data['number_of_trials'][0][0][0]
    trial_length = data['trial_length'][0][0][0][0]
    #make dict to relate trials # to neuron - - not clear that this needs to happen
    time_before = 400
    trial_length = 1600
    time_after = 400
    total_after = trial_length + time_after
    # times that currently don't get used

    time_bin = 1

    # vect_function = np.vectorize(bin_spikes)
    # binned_spikes = vect_function(time_bin, spikes, num_trials)

    binned_spikes = bin_spikes(time_bin, spikes, num_trials)

    # binned_spikes = np.zeros((500, 2435, int(1600/time_bin))) #need to add another dimension to average o ver

    # for i in range(spikes.shape[0]):
    #     for j in range(num_trials[i]):
    #         if spikes[i][j].size > 0:
    #             for k in spikes[i][j]:
    #                 temp_spike = int(k[0])

    #                 if temp_spike < 1600:
    #                     binned = int((temp_spike/time_bin))
    #                     binned_spikes[i][j][binned] = 1

    #             #print (np.array_equal(binned_spikes_test, binned_spikes[i][j]))
                
    # binned_spikes_test = list(filter(lambda x: x<=1600, spikes[:, :, :, 0]))

    #binned_spikes2 = list(map(conv_spikes, binned_spikes_test[:,:]))
    #list(filter(lambda x,y: x[y][0] <=1600, spikes[:][:])
            


    summed_spikes = np.zeros((500, int(1600/time_bin)))

    # for c in range(binned_spikes.shape[0]):
    for c in range(1):

        for i in range(binned_spikes.shape[2]): #per time slice
            for j in range(binned_spikes.shape[1]): #per trial
            # print (i,j)
                summed_spikes[c][i] = summed_spikes[c][i] + \
                    binned_spikes[c][j][i]


    # plt.plot(scipy.ndimage.gaussian_filter(sum, 45))
    # plt.show()

    return (spikes, conditions, num_trials, trial_length, summed_spikes, binned_spikes)


def bin_spikes(time_bin, spikes, num_trials):
    binned_spikes = np.zeros((500, 2435, int(1600/time_bin))) #need to add another dimension to average o ver

    for i in range(spikes.shape[0]):
        for j in range(num_trials[i]):
            if spikes[i][j].size > 0:
                for k in spikes[i][j]:
                    temp_spike = int(k[0])

                    if temp_spike < 1600:
                        binned = int((temp_spike/time_bin))
                        binned_spikes[i][j][binned] = 1

    return binned_spikes

def bin_spikes_improved(time_bin, spikes, num_trials):

    
    return 1


