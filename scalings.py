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


class sigmoid_scale:
    def __init__(self, maximize=True, a=3):
        self.maximize = maximize
        self.a = a

    def scale(self, values):
        sigma = np.median(values)
        sig_scale = (1 + ((values/sigma)**self.a))**-1
        if self.maximize:
            sig_scale = 1 - sig_scale
        return sig_scale

