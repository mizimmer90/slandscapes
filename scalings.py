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
    def __init__(self, maximize=True,a=3,b=2):
        self.maximize = maximize
        self.a = a
        self.b = b
    def scale(self, values):
        sigma = np.median(values)
        if self.maximize:
            scaled_values = (1 - (1+(((2**(self.a/self.b)) - 1)*((values/sigma)**self.a)))**(-self.b/self.a))
        else:
            scaled_values = 1-(1 - (1+(((2**(self.a/self.b)) - 1)*((values/sigma)**self.a)))**(-self.b/self.a))
        return scaled_values



