from analysis import AnalyzeCell
import models
import time
import matplotlib.pyplot as plt 
from cellplot import CellPlot
import numpy as np
import os
import datetime
import json


class AnalysisPipeline(object):

    """Performs fitting procedure and model comparison for all cells.

    Parameters
    ----------
    no_cells : int
        Integer signifying the number of cells to be analyzed.
    data_processor : DataProcessor
        Object returned by data processing module that includes all relevent cell data.
    models : list of str
        List of model names as strings to be used in analysis.
        These must match the names used in "fit_x" methods.

    Attributes
    ----------
    no_cells : int
        Integer signifying the number of cells to be analyzed.
    data_processor : DataProcessor
        Object returned by data processing module that includes all relevent cell data.
    models : list of str
        List of model names as strings to be used in analysis.
        These must match the names used in "fit_x" methods.
    time_info : TimeInfo
        Object that holds timing information including the beginning and end of the region
        of interest and the time bin. All in seconds.
    analysis_dict : dict (int: AnalyzeCell)
        Dict containing model fits per cell as contained in AnalyzeCell objects.
    model_fits : dict (str: dict (int: Model))
        Nested dictionary containing all model fits, per model per cell.

    """

    def __init__(self, cell_range, data_processor, models, subsample, swarm_params=None):
        self.time_start = time.time()
        self.cell_range = cell_range[:]
        self.cell_range[1] += 1
        self.data_processor = data_processor
        self.time_info = data_processor.time_info
        self.models_to_fit = models
        self.subsample = subsample
        if not swarm_params:
            self.swarm_params = {
                "phip" : 0.5,
                "phig" : 0.5,
                "omega" : 0.5,
                "minstep" : 1e-8,
                "minfunc" : 1e-8,
                "maxiter" : 1000
            }
        else:
            self.swarm_params = swarm_params
        self.analysis_dict = self.make_analysis()
        self.model_dict = self.make_models()
        self.subsample = subsample
        self.model_fits = None

    def make_analysis(self):
        analysis_dict = {}
        for cell in range(*self.cell_range):
            analysis_dict[cell] = AnalyzeCell(
                cell, self.data_processor, self.subsample)
        return analysis_dict

    def make_models(self):
        model_dict = {}
        for model in self.models_to_fit:
            model_dict[model] = {}
        for cell in range(*self.cell_range):
            for model in self.models_to_fit:

                #data passed here is manually selected by what models need
                model_data = {}
                model_data['spikes_time'] = self.analysis_dict[cell].time_spikes_binned
                model_data['time_info'] = self.analysis_dict[cell].time_info
                model_data['num_trials'] = self.analysis_dict[cell].num_trials
                model_data['conditions'] = self.analysis_dict[cell].conditions
                model_data['spikes_pos'] = self.analysis_dict[cell].position_spikes_binned
                model_data['pos_info'] = self.analysis_dict[cell].pos_info
                model_data['swarm_params'] = self.swarm_params
                #this creates an instance of class "model" in the module "models"
                model_instance = getattr(models, model)(model_data)

                model_dict[model][cell] = model_instance
        return model_dict

    def set_model_bounds(self, model, bounds):
        if model in self.model_dict:
            for cell in range(*self.cell_range):
                self.model_dict[model][cell].set_bounds(bounds)
        else:
            raise ValueError("model does not match supplied models")

    def fit_all_models(self, iterations):
        cell_fits = {}
        for cell in range(*self.cell_range):
            cell_fits[cell] = {}
            for model in self.model_dict:
                model_instance = self.model_dict[model][cell]
                getattr(self.analysis_dict[cell], "fit_model")(model_instance, iterations)
                cell_fits[cell][model_instance.__class__.__name__] = model_instance.fit.tolist()
        self.format_save(cell_fits)
        # np.save(os.getcwd() + "/results/cell_fits_" +
        #     str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.'), cell_fits)

    def format_save(self, fits):
        output = fits

        # for cell in fits:
        #     output[cell] = {}
        #     for model in fits[cell]:
        #         output[cell][model] = 
        #         for ind, value in enumerate(fits[cell][model]):
        #             output[cell][model][ind] = value
        
        # for index, param in enumerate(model.params):
        #     output[cell][name][param] = model.fit[index]
                    #     'a_1' : fit[cell][model][0],
                    #     'mu' : fit[cell][model][1],
                    #     'sd' : fit[cell][model][2],
                    #     'o' : fit[cell][model][3]
        #             # }
        # return output
        with open(os.getcwd() + "/results/cell_fits.txt", 'a') as out:
            json.dump(output, out)

    def save_comparison(self, comp):
        # output = {}
        # for cell in comp:
        #     output[cell] = {
        #         ''
        #     }
        with open(os.getcwd() + "/results/model_comparisons.txt", 'a') as out:
            json.dump(comp, out)

    def compare_models(self, model_min, model_max):
        outcomes = {}
        for cell in range(*self.cell_range):
            plotter = CellPlot(self.analysis_dict[cell])
            min_model = self.model_dict[model_min][cell]
            max_model = self.model_dict[model_max][cell]
            print(min_model.fit)
            print(max_model.fit)
            outcome = str(self.analysis_dict[cell].compare_models(
                    min_model, 
                    max_model
            ))
            print(outcome)
            outcomes[cell] = outcome
            plotter.plot_comparison(min_model, max_model)
            print("TIME IS")
            print(time.time() - self.time_start)
            plt.show()
        # np.save(os.getcwd() + "/results/comparison_"+model_max+"_" + model_min + 
        #     "_" + str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.'), outcomes)
        self.save_comparison(outcomes)

    def show_condition_fit(self, model):
        for cell in range(*self.cell_range):
            plotter = CellPlot(self.analysis_dict[cell])
            extracted_model = self.model_fits[model][cell]
            plotter.plot_cat_fit(extracted_model)
            plt.show()


