import numpy as np
import matplotlib.pyplot as plt
import parse_matlab_data
import time
import pandas
import time_info
import os
import math


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
    binned_spikes : dict (int: numpy.ndarray)
        Dict containing binned spike data for all cells, indexed by cell.
    conditions_dict : dict (tuple of int, int: numpy.ndarray of int)
        Dict containing condition information for all cells.
        Indexed by a tuple of format (condition, cell number).

    """

    def __init__(self, path, time_info, num_cells, num_conditions=0):
        self.path = path
        self.num_cells = num_cells
        # self.time_high_ms = time_info.time_high * 1000
        # self.time_low_ms = time_info.time_low * 1000
        # self.time_bin_ms = time_info.time_bin * 1000
        self.time_high_ms = time_info.time_high 
        self.time_low_ms = time_info.time_low 
        self.time_bin_ms = time_info.time_bin   
        self.time_info = time_info
        self.num_conditions = num_conditions
        self.spikes = self.extract_spikes("ms")
        self.num_trials = self.extract_num_trials()
        self.conditions = self.extract_conditions()
        self.binned_spikes = self.bin_spikes()
        self.summed_spikes = self.sum_spikes()
        self.conditions_dict = self.associate_conditions()
        self.summed_spikes_condition = self.sum_spikes_conditions()
        self.position_data = self.extract_position()
        self.binned_position = self.bin_position()
    

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
        print(spikes[0][0])
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
            # print(self.binned_spikes[cell])
            # print(np.sum(self.binned_spikes[cell], 1))
            summed_spikes[cell] = np.sum(self.binned_spikes[cell], 0)
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
                        self.binned_spikes[cell].T * self.conditions_dict[condition + 1, cell], 1)

            return summed_spikes_condition

    def bin_spikes(self):
        """Bins spike data into configurable time bins.

        Returns
        -------
        dict (int: numpy.ndarray of int)
            Binned spike data per cell.

        """
        binned_spikes = {}
        for cell in range(self.num_cells):
            binned_spikes[cell] = np.zeros((self.num_trials[cell], self.time_info.total_bins))

            for trial_index, trial in enumerate(self.spikes[cell]):
                if type(trial) is np.ndarray:
                    for time in trial:

                        if time < self.time_high_ms and time >= self.time_low_ms:
                            binned_spikes[cell][trial_index][int(time - self.time_low_ms)] = 1
                        
        return binned_spikes

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

    def save_binned_spikes(self):
        """Saves binned spike data to disk.

        """
        np.save("binned_spikes", self.binned_spikes)
