import numpy as np


class feature_scale:
    def __init__(self, maximize=True):
        self.maximize = maximize

    def scale(self, values):
        value_spread = values.max() - values.min()
        if self.maximize:
            scaled_values = (values - values.min()) / value_spread
        else:
            scaled_values = (values.max() - values) / value_spread
        return scaled_values
