import numpy as np
import matplotlib.pyplot as plt
import parse_matlab_data
import time
import pandas


def parse_python(num_cells, time_range, time_bin):
    path_to_data = '/Users/stevecharczynski/workspace/python_ready_data'
    time_range_ms = np.multiply(time_range , 1000)
    time_bin_ms = time_bin * 1000
    num_time_points = (time_range[1] - time_range[0]) / time_bin
    spikes = {}
    for i in range(num_cells):
        spike_path = path_to_data + '/spikes/%d.npy' % i
        spikes[i] = np.load(spike_path, encoding="bytes")

    num_trials = np.loadtxt(path_to_data + '/number_of_trials.csv', delimiter=',', dtype='int')
    conditions  = np.loadtxt(path_to_data + '/conditions.csv', delimiter=',', dtype='int')
    spikes = {k: v * 1000 for k, v in spikes.items()}
    # trick to pull out summed spikes, may make more straightforward in the future
    summed_spikes = {}
    for i in range(num_cells):
        flat_spikes = np.hstack(spikes[i]).astype(int)
        flat_spikes = flat_spikes[flat_spikes<=2000]
        flat_spikes = flat_spikes[flat_spikes>=400]

        unique, counts = np.unique(flat_spikes, return_counts = True)
        summed_spikes[i] = counts
    # creating binned spikes array
    binned_spikes = {}
    #binned_spikes = np.zeros((num_cells, num_trials[0], int(time_range_ms[1]-time_range_ms[0]/time_bin_ms)))
    for cell in range(num_cells):
        binned_spikes[cell] = np.zeros((num_trials[cell], int(time_range_ms[1]-time_range_ms[0]/time_bin_ms)))
        #binned_spikes[cell] = np.zeros((num_trials[cell], 2400))

        for trial_index, trial in enumerate(spikes[cell]):
            for time in trial:
                if time < time_range_ms[1] and time >= time_range_ms[0]:
                    binned_spikes[cell][trial_index][int(time-time_range_ms[0])] = 1
                    #binned_spikes[cell][trial_index][int(time)] = 1

    sum_spike_new = {}
    for cell in range(num_cells):
        sum_spike_new[cell] = np.zeros((int(time_range_ms[1]-time_range_ms[0]/time_bin_ms)))
        for trial in range(num_trials[cell]):
            for time in range(binned_spikes[cell][trial].shape[0]):
                sum_spike_new[cell][time] += binned_spikes[cell][trial][time]
    conditions_dict = {}
    # associating trial and conditions
    for cell in range(num_cells):
        cond = conditions[cell][0:num_trials[cell]]
        conditions_dict[1, cell] = np.zeros([num_trials[cell]])
        conditions_dict[2, cell] = np.zeros([num_trials[cell]])
        conditions_dict[3, cell] = np.zeros([num_trials[cell]])
        conditions_dict[4, cell] = np.zeros([num_trials[cell]])

        for trial, condition in enumerate(cond):
            if condition:
                conditions_dict[condition, cell][trial] = 1

    return sum_spike_new, binned_spikes, conditions_dict, spikes

  

