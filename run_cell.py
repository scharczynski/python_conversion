import sys
from describe_data import DescribeData
from python_data import DataProcessor
from region_info import RegionInfo
from analysis_pipeline import AnalysisPipeline


cell_range = sys.argv[-2:]
cell_range = list(map(int, cell_range))
# print((cell_range))


# path_to_data = '/Users/stevecharczynski/workspace/brincat_miller'
path_to_data = "/usr3/bustaff/scharcz/workspace/brincat_miller/"
time_info = RegionInfo(500, 1750, 1.0)
data_descriptor = DescribeData(path_to_data, False, "ms", cell_range, 0, time_info=time_info)
data_processor = DataProcessor(data_descriptor)

n = 2
mean_delta = 0.10 * (time_info.region_high - time_info.region_low)
mean_bounds = (
    (time_info.region_low - mean_delta),
    (time_info.region_high + mean_delta))

bounds = ((0.001, 1 / n), mean_bounds, (10, 1000), (10**-10, 1 / n))
bounds_c = (10**-10, 0.99)
pipeline = AnalysisPipeline(cell_range, data_processor, ["Const", "Time"], 0)
pipeline.set_model_bounds("Time", bounds)
pipeline.set_model_bounds("Const", bounds_c)
pipeline.fit_all_models()
pipeline.compare_models("Const", "Time")