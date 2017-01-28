import collections
import itertools
import numpy as np
import pylab as plt

def _gaussian(xs, a, b, c):
    f = a*np.exp(-((xs-b)**2) / (2 * (c**2)))
    return f

def _gaussian_multD(xs, x0s, height=1, widths=1, normed=False):
    # check dim of a
    if isinstance(height, collections.Iterable):
        raise
    # check dim of x0s
    if isinstance(x0s, collections.Iterable):
        if len(x0s) != len(xs):
            raise
    else:
        x_shape = xs.shape
        if len(x_shape) >= 3:
            x0s = np.array([x0s for i in range(x_shape[0])])
    # check dim of c
    if isinstance(widths, collections.Iterable):
        if len(widths) != len(xs):
            raise
    else:
        x_shape = xs.shape
        if len(x_shape) >= 3:
            widths = np.array([widths for i in range(x_shape[0])])
    # calc exponent
    exponent = np.sum(
        [
            ((xs[num] - x0s[num])**2) / (2 * (widths[num]**2))
            for num in range(len(xs))],
        axis=0)
    if normed:
        height = np.prod(1/(((2*np.pi)**0.5)*np.array(widths)))*height
    f = height*np.exp(-exponent)
    return f

def gaussian_noise(
        x1s, x2s, gaussians_per_axis=4, height_range=[-0.1,0.1],
        width_range=[0.9,1.1], rigidity=0):
    """Given x1 coords and x2 coords, generates a specified number of evenly
       spaced gaussians along each axis of varying height and widths. The
       rigidity value determines how evenly spaced gaussians are (0 is loose
       and 1 is rigid)."""
    # dimension info
    axis1 = x1s[0]
    axis2 = x2s[:,0]
    box_length_1 = (axis1[-1]-axis1[0])/gaussians_per_axis
    box_length_2 = (axis2[-1]-axis2[0])/gaussians_per_axis
    box_centers_1 = (np.arange(gaussians_per_axis)*box_length_1) + (box_length_1/2.0)
    box_centers_2 = (np.arange(gaussians_per_axis)*box_length_2) + (box_length_2/2.0)
    # generate rigid center coords
    centers = np.array(
        list(itertools.product(box_centers_1,box_centers_2)))
    # heights
    height_spread = height_range[1] - height_range[0]
    heights = [
        height_spread * np.random.random() + height_range[0]
        for n in range(gaussians_per_axis**2)]
    # widths
    widths_spread = width_range[1] - width_range[0]
    widths = [
        widths_spread * np.random.random() + width_range[0]
        for n in range(gaussians_per_axis**2)]
    # rigid formula and initialize noise and xs
    rigidity_div = 2 + (rigidity*4)**2
    noise = np.zeros((len(axis1), len(axis2)))
    xs = np.array([x1s, x2s])
    # add gaussians
    for num in range(gaussians_per_axis**2):
        rand1 = np.random.random()-0.5
        rand2 = np.random.random()-0.5
        center = [
            centers[num, 0] + (box_length_1/rigidity_div)*rand1,
            centers[num, 1] + (box_length_2/rigidity_div)*rand2]
        noise += _gaussian_multD(
            xs, center, height=heights[num], widths=widths[num])
    return noise

def surface_to_probs(x1s, x2s, surface, grid_size, adjust_centers=True):
    """given a potential energy landscape (in the form of values for x1,
       x2, and f(x1,x2) and a connectivity grid size) returns the transition
       probability matrix that corresponds to that surface. Looks for the
       highed energy between adjacent states and uses the arrhenius equation
       to generate a rate. Potential energy surface is in units of kT."""
    # identify number of states and resolution (points between states)
    n_states = grid_size[0]*grid_size[1]
    states = np.arange(n_states).reshape(grid_size)
    res = len(x1s[0])//grid_size[0]
    if adjust_centers:
        res_adjust = res//2
    else:
        res_adjust = 0
    # initialize empy probability matrix
    T = np.zeros((n_states, n_states))
    # Get transitions between columns on the grid
    for col in range(len(states[0])-1):
        # determine the maximum energy between column-adjacent states
        surface_slice = surface[
                res_adjust::res,
                (res_adjust + col*res):((col + 1)*res + 1 + res_adjust)]
        max_energy = np.max(surface_slice, axis=1)
        energy_diffs_1 = max_energy - surface_slice[:,0] 
        energy_diffs_2 = max_energy - surface_slice[:,-1]
        # convert energy diffs to rates
        rates1 = np.exp(-energy_diffs_1)
        rates2 = np.exp(-energy_diffs_2)
        # get state indices of transitions
        state_trans = states[:,col:col+2].T
        trans_iis_1 = (state_trans[0], state_trans[1])
        trans_iis_2 = (state_trans[1], state_trans[0])
        T[trans_iis_1] = rates1
        T[trans_iis_2] = rates2
    # Get transitions between rows on the grid
    for row in range(len(states)-1):
        # determine the maximum energy between row-adjacent states
        surface_slice = surface[
                (res_adjust + row*res):((row + 1)*res + 1 + res_adjust),
                res_adjust::res]
        max_energy = np.max(surface_slice, axis=0)
        energy_diffs_1 = max_energy - surface_slice[0, :]
        energy_diffs_2 = max_energy - surface_slice[-1, :]
        # convert energy diffs to rates
        rates1 = np.exp(-energy_diffs_1)
        rates2 = np.exp(-energy_diffs_2)
        # get state indices of transitions
        state_trans = states[row:row+2,:]
        trans_iis_1 = (state_trans[0], state_trans[1])
        trans_iis_2 = (state_trans[1], state_trans[0])
        T[trans_iis_1] = rates1
        T[trans_iis_2] = rates2
    # Get diagonal transitions
    T[range(len(T)), range(len(T))] = 1
    # normalize rows and return
    T /= T.sum(axis=1)[:,None]
    return T

def funneled_landscape(grid_size, width_frac=0.707, depth=1, resolution=1):
    l = landscape(grid_size=grid_size, resolution=resolution)
    x0 = np.array([grid_size[0], grid_size[1]])
    widths = x0*width_frac
    l = l.add_gaussian(x0, height=-depth, widths=widths)
    return l

def diagonal_barrier(
        grid_size, position=0.5, height=1, width=1, resolution=1):
    l = landscape(grid_size=grid_size, resolution=resolution)
    semi_circum = len(l.x1_coords)+len(l.x1_coords[0]) - 1
    diag = int(semi_circum * position) - len(l.x1_coords)
    centers = np.array(
        list(
            zip(
                l.x1_coords[::-1].diagonal(diag),
                l.x2_coords[::-1].diagonal(diag))))
    for center in centers:
        l = l.add_gaussian(x0s=center, height=height, widths=width)
    return l

def egg_carton_landscape(
        grid_size, gaussians_per_axis, height=1, width=1, resolution=1):
    l = landscape(grid_size=grid_size, resolution=resolution)
    l = l.add_noise(
        gaussians_per_axis=gaussians_per_axis, height_range=[height, height],
        width_range=[width, width], rigidity=100000)
    return l

class landscape:
    def __init__(
            self, grid_size,  x1_coords=None, x2_coords=None, values=None,
            resolution=1):
        if (x1_coords is None) and (x2_coords is None) and (values is None):
            if (grid_size is None):
                print("Need to specify a grid_size")
                raise
            self.grid_size = grid_size
            x1_points = np.arange(grid_size[0]*resolution)/resolution
            x2_points = np.arange(grid_size[1]*resolution)/resolution
            self.x1_coords, self.x2_coords = np.meshgrid(x1_points, x2_points)
            self.values = np.zeros(self.x1_coords.shape)
        else:
            if (x1_coords is None) or (x2_coords is None) or (values is None):
                print("missing inputs")
                raise
            self.grid_size = (
                int(x1_coords[0][-1] + x1_coords[0][1]),
                int(x2_coords[:,0][-1] + x2_coords[:,0][1]))
            self.x1_coords = x1_coords
            self.x2_coords = x2_coords
            self.values = values
    def add_gaussian(self, x0s, height=1, widths=1):
        input_coords = np.array([self.x1_coords, self.x2_coords])
        new_values = self.values + _gaussian_multD(
            input_coords, x0s, height=height, widths=widths)
        return landscape(
            grid_size=self.grid_size, x1_coords=self.x1_coords,
            x2_coords=self.x2_coords, values=new_values)
    def add_noise(
            self, gaussians_per_axis, height_range=[-0.1, 0.1],
            width_range=[0.85, 1.15], rigidity=0):
        noise = gaussian_noise(
            self.x1_coords, self.x2_coords,
            gaussians_per_axis = gaussians_per_axis, height_range = height_range,
            width_range = width_range, rigidity = rigidity)
        return landscape(
            grid_size=self.grid_size, x1_coords=self.x1_coords,
            x2_coords=self.x2_coords, values=self.values+noise)
    def plot(self, title='potential energy landscape'):
        plt.figure(title)
        plt.xlim((self.x1_coords[0,0], self.x1_coords[0,-1]))
        plt.ylim((self.x2_coords[0,0], self.x2_coords[-1,0]))
        plt.pcolormesh(self.x1_coords, self.x2_coords, self.values)
        plt.colorbar()
        plt.show()
        return
    def to_probs(self):
        T = surface_to_probs(
            self.x1_coords, self.x2_coords, self.values, self.grid_size)
        return T

