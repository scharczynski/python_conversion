class TimeInfo(object):
    def __init__(self, time_low, time_high, time_bin):
        self.time_low = time_low
        self.time_high = time_high
        self.time_bin = time_bin
        self.total_bins = (self.time_high - self.time_low) / (self.time_bin)
