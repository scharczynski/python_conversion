class DescribeData(object):

    def __init__(self, path, has_position, time_units, num_cells, num_conditions, **regions):

        self.path = path
        self.has_position = has_position
        self.time_units = time_units
        self.num_cells = num_cells
        self.num_conditions = num_conditions

        if "time_info" in regions.keys():
            self.time_info = regions["time_info"]
        else:
            self.time_info = None

        if "pos_info" in regions.keys():
            self.pos_info = regions["pos_info"]
        else:
            self.pos_info = None
