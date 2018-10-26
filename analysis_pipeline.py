from analysis import AnalyzeCell
import models
import time
import matplotlib.pyplot as plt 
from cellplot import CellPlot


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

    def __init__(self, no_cells, data_processor, models, subsample):
        self.time_start = time.time()
        self.no_cells = no_cells
        self.data_processor = data_processor
        self.time_info = data_processor.time_info
        self.models = models
        self.subsample = subsample
        self.analysis_dict = self.make_analysis()
        self.subsample = subsample
        self.model_fits = None

    def make_analysis(self):
        analysis_dict = {}
        for cell in range(self.no_cells):
            analysis_dict[cell] = AnalyzeCell(
                cell, self.data_processor, self.subsample)
        return analysis_dict

    def fit_all(self):
        model_fits = {}
        for model in self.models:
            model_fits[model] = {} 

        for cell in range(self.no_cells):
            for model in self.models:
                # model_fits[model][cell] = getattr(
                #     self.analysis_dict[cell], "fit_" + model)()

                model_data = {}
                model_data['spikes'] = self.analysis_dict[cell].binned_spikes
                model_data['time_info'] = self.analysis_dict[cell].time_info
                model_data['num_trials'] = self.analysis_dict[cell].num_trials
                model_data['conditions'] = self.analysis_dict[cell].conditions
                model_data['position'] = self.analysis_dict[cell].binned_position

                model_instance = getattr(models, model)(model_data)
                model_fits[model][cell] = getattr(self.analysis_dict[cell], "fit_model")(model_instance)
        self.model_fits = model_fits
        return model_fits

    def compare_models(self, model_min, model_max):
        for cell in range(self.no_cells):
            plotter = CellPlot(self.analysis_dict[cell])
            min_model = self.model_fits[model_min][cell]
            max_model = self.model_fits[model_max][cell]
            print(min_model.fit)
            print(max_model.fit)
            print(
                self.analysis_dict[cell].compare_models(
                    min_model, max_model))
            plotter.plot_comparison(min_model, max_model)
            plt.show()
            #self.analysis_dict[cell].plot_cat_fit(max_model)
            # plotter.plot_cat_fit(max_model)

        print("TIME IS")
        print(time.time() - self.time_start)
