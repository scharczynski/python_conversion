class RegionInfo(object):
    def __init__(self, region_low, region_high, region_bin):
        self.region_low = region_low
        self.region_high = region_high
        self.region_bin = region_bin
        self.total_bins =round((self.region_high- self.region_low) / (self.region_bin))
