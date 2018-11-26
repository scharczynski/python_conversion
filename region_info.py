class RegionInfo(object):
    def __init__(self, region_low, region_high, region_bin):
        self.region_low = region_low
        self.region_high = region_high
        self.region_bin = region_bin
        self.total_bins = self.calc_bins()
        self.converted = False

    def to_ms(self):
        self.region_low *= 1000
        self.region_high *= 1000
        self.region_bin *= 1000
        self.converted = True

    def calc_bins(self):
        return round((self.region_high- self.region_low) / (self.region_bin))