import numpy as np
import matplotlib.pyplot as plt
import parse_matlab_data
import time
import pandas
import time_info
import os
import math
from describe_data import DescribeData


class DataProcessor(object):

    """Extracts data from given python-friendly formatted dataset.

    Parameters
    ----------
    path : str
        Path to data files. In current form, this is a parent folder
        containing other data files.
    time_info : TimeInfo
        Object that holds timing information including the beginning and
        end of the region of interest and the time bin. All in seconds.
    num_cells : int
        Integer signifying the number of cells in the dataset.
    num_conditions : int
        Integer signifying the number of experimental conditions in the
        dataset.

    Attributes
    ----------
    path : str
        Path to data files. In current form, this is a parent folder
        containing other data files.
    time_high_ms : int
        End of region of interest in milliseconds.
    time_low_ms : int
        Beginning of region of interest in milliseconds.
    time_bin_ms : int
        Size of time bin in milliseconds.
    time_info : TimeInfo
        Object that holds timing information including the beginning and
        end of the region of interest and the time bin. All in seconds.
    num_cells : int
        Integer signifying the number of cells in the dataset.
    num_conditions : int
        Integer signifying the number of experimental conditions in the
        dataset.
    spikes : numpy.ndarray
        Array of spike times in milliseconds, of dimension (trials × time).
    num_trials : numpy.ndarray
        Array containing integers signifying the number of trials a given cell has data for.
    conditions : numpy.ndarray
        Array containing condition data (integers 1 through 4) per cell per trial.
    summed_spikes : dict (int: numpy.ndarray)
        Dict containing summed spike data for all cells, indexed by cell.
    time_spikes_binned : dict (int: numpy.ndarray)
        Dict containing binned spike data for all cells, indexed by cell.
    conditions_dict : dict (tuple of int, int: numpy.ndarray of int)
        Dict containing condition information for all cells.
        Indexed by a tuple of format (condition, cell number).

    """

    def __init__(self, data_descriptor):
        self.path = data_descriptor.path
        self.num_cells = data_descriptor.num_cells
        self.num_conditions = data_descriptor.num_conditions
        self.spikes = self.extract_spikes(data_descriptor.time_units)
        self.num_trials = self.extract_num_trials()
        self.conditions = self.extract_conditions()
        self.position_data = self.extract_position()

        self.conditions_dict = self.associate_conditions()


        if data_descriptor.time_info is not None:
            self.time_info = data_descriptor.time_info
            self.time_high_ms = data_descriptor.time_info.region_high 
            self.time_low_ms = data_descriptor.time_info.region_low 
            self.time_bin_ms = data_descriptor.time_info.region_bin
            self.total_time_bins = data_descriptor.time_info.total_bins
            if data_descriptor.time_units == "s":
                self.time_high_ms *= 1000
                self.time_low_ms *= 1000
                self.time_bin_ms *= 1000

            self.time_spikes_binned = self.bin_spikes_time()
            self.summed_spikes = self.sum_spikes()


        if data_descriptor.pos_info is not None:
            self.pos_info = data_descriptor.pos_info
            self.pos_high = data_descriptor.pos_info.region_high 
            self.pos_low = data_descriptor.pos_info.region_low 
            self.pos_bin = data_descriptor.pos_info.region_bin  
            self.total_pos_bins = data_descriptor.pos_info.total_bins



        self.summed_spikes_condition = self.sum_spikes_conditions()
        self.position_spikes_binned = self.bin_spikes_position()
        self.summed_position_spikes = self.sum_position_spikes()
    

    def extract_spikes(self, units):
        """Extracts spike times from data file and converts to miliseconds.

        Returns
        -------
        dict (int: numpy.ndarray of float?)
            Contains per cell spike times.

        """
        spikes = {}
        if os.path.exists(self.path + "/spikes/"):
            for i in range(self.num_cells):
                spike_path = self.path + '/spikes/%d.npy' % i
                spikes[i] = np.load(spike_path, encoding="bytes")
            if units == "s":
                spikes = {k: v * 1000 for k, v in spikes.items()}
        else:
            print("Spikes folder not found.")
        return spikes

    def extract_num_trials(self):
        """Extracts number of trials per cell.

        Returns
        -------
        numpy.ndarray of int
            Array of dimension [Number of cells] that provides the number of
            trials of a given cell.

        """
        if os.path.exists(self.path + "/number_of_trials.csv"):
            return np.loadtxt(
                self.path +
                '/number_of_trials.csv',
                delimiter=',',
                dtype='int')
        else:
            print("number_of_trials.csv not found")
            return None

    def extract_conditions(self):
        """Extracts trial conditions per cell per trial.

        Returns
        -------
        numpy.ndarray of int
            Array of dimension [Number of cells] × [Trials] that provides the condition
            for the trial.

        """
        if os.path.exists(self.path + "/conditions.csv"):
            return np.loadtxt(
                self.path +
                '/conditions.csv',
                delimiter=',',
                dtype='int')
        else:
            print("conditions.csv not found")
            return None

    def extract_position(self):
        if os.path.exists(self.path + "/position/"):
            return np.load(
                self.path + "/position/xy_data.npy",
                encoding="bytes"
            )
        else:
            print("xy_data.npy not found")
            return None

    def sum_spikes(self):
        """Sums spike data over trials.

        Returns
        -------
        dict (int: numpy.ndarray of int)
            Summed spike data.

        Todo:
        ----
        It may be correct here to normalize against number of trials.

        """
        summed_spikes = {}
        for cell in range(self.num_cells):
            # print(self.time_spikes_binned[cell])
            # print(np.sum(self.time_spikes_binned[cell], 1))
            summed_spikes[cell] = np.sum(self.time_spikes_binned[cell], 0)
        return summed_spikes

    def sum_spikes_conditions(self):
        """Sums spike data over trials per condition

        Returns
        -------
        numpy.ndarray of int
            Array of dimension [Cells] × [Condition] × [Time].

        Todo
        ----
        Might be worth condensing this method with sum_spikes

        """
        if self.conditions is not None:
            total_time_bins = int(
                (self.time_high_ms -
                self.time_low_ms) /
                self.time_bin_ms)
            summed_spikes_condition = np.zeros(
                (self.num_cells, self.num_conditions, total_time_bins))

            for cell in range(self.num_cells):
                for condition in range(self.num_conditions):
                    summed_spikes_condition[cell][condition] = np.sum(
                        self.time_spikes_binned[cell].T * self.conditions_dict[condition + 1, cell], 1)

            return summed_spikes_condition

    def bin_spikes_time(self):
        """Bins spike data into configurable time bins.

        Returns
        -------
        dict (int: numpy.ndarray of int)
            Binned spike data per cell.

        """
        time_spikes_binned = {}
        for cell in range(self.num_cells):
            time_spikes_binned[cell] = np.zeros((self.num_trials[cell], self.total_time_bins))

            for trial_index, trial in enumerate(self.spikes[cell]):
                if type(trial) is np.ndarray:
                    for time in trial:

                        if time < self.time_high_ms and time >= self.time_low_ms:
                            time_spikes_binned[cell][trial_index][int(time - self.time_low_ms)] = 1
                        
        return time_spikes_binned

    def associate_conditions(self):
        """Builds dictionary that associates trial and condition.

        Returns
        -------
        dict (int, int: np.ndarray of int)
            Dict indexed by condition AND cell number returning array of trials.
            Array contains binary data: whether or not the trial is of the indexed condition.

        """
        if self.conditions is not None:
            conditions_dict = {}
            for cell in range(self.num_cells):
                cond = self.conditions[cell][0:self.num_trials[cell]]
                for i in range(self.num_conditions):
                    conditions_dict[i + 1, cell] = np.zeros([self.num_trials[cell]])
                for trial, condition in enumerate(cond):
                    if condition:
                        conditions_dict[condition, cell][trial] = 1
            return conditions_dict

    def bin_position(self):
        if self.position_data is None:
            return None
        binned_x = np.zeros((self.time_info.total_bins))
        binned_y = np.zeros((self.time_info.total_bins))
        for index, t in enumerate(self.position_data[:,0]):
            time_ind = int(round(t))
            if not binned_x[time_ind]:
                binned_x[time_ind] = self.position_data[index, 1]
                binned_y[time_ind] =  self.position_data[index, 2]
        return (binned_x, binned_y)

    def save_time_spikes_binned(self):
        """Saves binned spike data to disk.

        """
        np.save("time_spikes_binned", self.time_spikes_binned)

    def bin_spikes_position(self):
        spike_pos_cells = {}
        max_trial = 0
        for cell in self.spikes.keys():
            for i in self.spikes[cell]:
                if len(i) > max_trial:
                    max_trial = len(i)
            
        for cell in range(self.num_cells):
            spike_pos_cells[cell] = np.zeros((self.num_trials[cell], max_trial+1))
            for trial in range(self.num_trials[cell]):
                spike_count = 0
                for spike in self.spikes[cell][trial]:
                    spike_count += 1
                    index = int(spike * 1000)
                    pos_time = int(self.position_data[:,0].flat[np.abs(self.position_data[:,0] - index).argmin()])
                    pos_index = np.where(self.position_data[:,0] == pos_time)
                    spike_pos_x = self.position_data[pos_index[0], 1][0]
                    spike_pos_cells[cell][trial][spike_count] = spike_pos_x

        min_x = min(self.position_data[:,1])
        min_y = min(self.position_data[:,2])
        max_x = max(self.position_data[:,1])
        max_y = max(self.position_data[:,2])

        position_spike = {}
        for cell in range(self.num_cells):
            position_spike[cell] = np.zeros((self.num_trials[cell], int(max_x+1)))
            for trial in range(self.num_trials[cell]):
                for spike_pos in spike_pos_cells[cell][trial]:
                    if np.abs(spike_pos) > 0 :
                        position_spike[cell][trial][int(spike_pos)] += 1            

        return position_spike

    def sum_position_spikes(self):

        summed_position_spikes = {}
        for cell in range(self.num_cells):
            # print(self.time_spikes_binned[cell])
            # print(np.sum(self.time_spikes_binned[cell], 1))
            summed_position_spikes[cell] = np.sum(self.position_spikes_binned[cell], 0)

        return summed_position_spikes
