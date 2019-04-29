import collections
import glob
import itertools
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import os
from mdtraj import io

#######################################################################
#                   formating and plotting stuff                      #
#######################################################################


def _convert_path(path, n_states):
    dim = int(np.sqrt(n_states))
    shape = (dim, dim)
    states_shape = np.arange(n_states).reshape(shape)
    xy = np.array(
        [(np.concatenate(np.where(states_shape==state))) for state in path]).T
    return xy[::-1]


def convert_paths(paths, n_states):
    xys = []
    for path in paths:
        xy = _convert_path(path, n_states)
        xys.append(xy)
    return xys


def format_line(array):
    return " ".join([str(i) for i in array])


def format_flux_paths(fluxes, paths, n_states):
    xys = convert_paths(paths, n_states)
    if len(fluxes) != len(paths):
       raise
    output_data = ""
    for num in range(len(xys)):
        output_line = str(fluxes[num])+"\n"+\
            format_line(xys[num][0])+"\n"+\
            format_line(xys[num][1])+"\n"
        output_data += output_line
    return output_data


def plot_pijs(filenames, grid_size=None, output_names=None, state_num=None):
    if isinstance(filenames, collections.Iterable) and not \
            isinstance(filenames, (str, bytes)):
        filenames = filenames
    else:
        filenames = glob.glob(filenames)
    data = [np.load(filename) for filename in filenames]
    if grid_size is None:
        grid_edge = np.sqrt(len(data[0]))
        grid_size = (grid_edge, grid_edge)
    X,Y = np.meshgrid(
        np.arange(grid_size[0]), np.arange(grid_size[1]))
    for num in range(len(filenames)):
        plt.figure(filenames[num])
        if state_num is None:
            plot_data = data[num].reshape(grid_size)
        else:
            plot_data = data[num][state_num].reshape(grid_size)
        plt.pcolormesh(X, Y, plot_data, vmin=0, vmax=1)
        plt.colorbar()
        if output_names is not None:
            plt.savefig(output_names[num])
    plt.show()
    return


#######################################################################
#                         custom gaussians                            #
#######################################################################


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
        x1s, x2s, gaussians_per_axis=None, height_range=[-0.1,0.1],
        width_range=[0.9,1.1], rigidity=0):
    """Given x1 coords and x2 coords, generates a specified number of evenly
       spaced gaussians along each axis of varying height and widths. The
       rigidity value determines how evenly spaced gaussians are (0 is loose
       and 1 is rigid)."""
    if type(gaussians_per_axis) is int:
        gaussians_per_axis = [gaussians_per_axis, gaussians_per_axis]
    elif gaussians_per_axis is None:
        gaussians_per_axis = [int(x1s.max()/2.), int(x2s.max()/2.)]
    tot_gaussians = np.prod(gaussians_per_axis)
    # dimension info
    axis1 = x1s[0]
    axis2 = x2s[:,0]
    box_length_1 = (axis1[-1]-axis1[0])/gaussians_per_axis[0]
    box_length_2 = (axis2[-1]-axis2[0])/gaussians_per_axis[1]
    box_centers_1 = (np.arange(gaussians_per_axis[0]) * box_length_1) + \
        (box_length_1/2.0)
    box_centers_2 = (np.arange(gaussians_per_axis[1]) * box_length_2) + \
        (box_length_2/2.0)
    # generate rigid center coords
    centers = np.array(
        list(itertools.product(box_centers_1, box_centers_2)))
    # heights
    height_spread = height_range[1] - height_range[0]
    heights = [
        height_spread * np.random.random() + height_range[0]
        for n in range(tot_gaussians)]
    # widths
    widths_spread = width_range[1] - width_range[0]
    widths = [
        widths_spread * np.random.random() + width_range[0]
        for n in range(tot_gaussians)]
    # rigid formula and initialize noise and xs
    rigidity_div = 2 + (rigidity * 4)**2
    noise = np.zeros((len(axis2), len(axis1)))
    xs = np.array([x1s, x2s])
    # add gaussians
    for num in range(tot_gaussians):
        rand1 = np.random.random() - 0.5
        rand2 = np.random.random() - 0.5
        center = [
            centers[num, 0] + (box_length_1 / rigidity_div) * rand1,
            centers[num, 1] + (box_length_2 / rigidity_div) * rand2]
        noise += _gaussian_multD(
            xs, center, height=heights[num], widths=widths[num])
    return noise


#######################################################################
#            converting landscape to a probability matrix             #
#######################################################################


def energies_to_probs(Aij, energies):
    """Given an adjacency matrix and a list of state energies,
    returns the corresponding transition probability matrix.
    """
    # test shape of Aij
    fd, sd = Aij.shape
    if fd != sd:
        print("Aij is not square!")
        raise
    # test values of Aij
    if len(np.where((Aij != 0)*(Aij != 1))[0]) > 0:
        print("Aij has elements that are not 0 or 1...")
        raise
    # initialize probs
    probs = np.zeros(Aij.shape)
    for state in range(len(probs)):
        # find where there is a defined transition in Aij
        transitions = np.where(Aij[state] == 1)[0]
        # calculates rate = min[1, e^(U1-U2)]
        rates = np.minimum(
            np.ones(len(transitions)),
            np.exp(energies[state] - energies[transitions]))
        probs[state, transitions] = rates
    # norms the rates into transition probs
    probs = probs / probs.sum(axis=1)[:, None]
    return probs


def surface_to_probs(x1s, x2s, surface, grid_size, adjust_centers=True):
    """given a potential energy landscape (in the form of values for x1,
       x2, and f(x1,x2) and a connectivity grid size) returns the transition
       probability matrix that corresponds to that surface. Looks for the
       highest energy between adjacent states and uses the arrhenius equation
       to generate a rate. Potential energy surface is in units of kT."""
    # identify number of states and resolution (points between states)
    n_states = grid_size[0]*grid_size[1]
    states = np.arange(n_states).reshape((grid_size[1],grid_size[0]))
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
        # by taking the maximum energy, we don't need to employ the
        # metropolis min criterion, since the maximum could be the state
        # in question (i.e. the rate will never be less than 1)
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


def gen_aij(grid_size):
    """Generate a lattice with an arbitrary shape. Returns state
    numberings and the adjacency matrix."""
    n_states = np.prod(grid_size)
    states = np.arange(n_states)
    l_states = states.reshape(grid_size)
    iis = np.array(
        [np.array(np.where(l_states==state)).flatten() for state in states])
    aij = np.zeros((n_states, n_states), dtype=int)
    for state in states:
        diffs = iis - iis[state]
        dists = np.einsum('ij,ij->i', diffs, diffs)
        aij[state, np.where(dists<=1)[0]] = 1
    return aij

#######################################################################
#                           helper function                           #
#######################################################################

# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def point_seg_dist(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if np.arccos(np.dot((P - A) / np.linalg.norm(P - A), (B - A) / np.linalg.norm(B - A))) > np.pi / 2:
        return np.linalg.norm(P - A)
    if np.arccos(np.dot((P - B) / np.linalg.norm(P - B), (A - B) / np.linalg.norm(A - B))) > np.pi / 2:
        return np.linalg.norm(P - B)
    return np.linalg.norm(np.cross(A-B, A-P))/np.linalg.norm(B-A)

def draw_line(angle,dist,x0,y0):
    angle_r = np.deg2rad(angle)
    slope = np.tan(angle_r)
    cos = 1 / np.sqrt(1+slope**2)
    sin = slope / np.sqrt(1+slope**2)
    if angle < 90 or angle > 270:
        x1 = x0 + dist*cos
        y1 = y0 + dist*sin
    elif angle > 90 and angle < 270:
        x1 = x0 - dist*cos
        y1 = y0 - dist*sin
    return x1, y1

#######################################################################
#                           landscape class                           #
#######################################################################


class landscape:
    """Generation of a toy energy landscape.

    Parameters
    ----------
    grid_size : tuple, defaut=None
        The dimensions of the 2d landscape. i.e. for a landscape that
        is 50x10, grid_size=(50,10).
    x1_coords : array, shape=(grid_size[1], grid_size[0]), default=None
        The x1 coordinates of a pre-made grid. Not needed for a grid
        being initialized.
    x2_coords : array, shape=(grid_size[1], grid_size[0]), default=None
        The x2 coordinates of a pre-made grid. Not needed for a grid
        being initialized.
    values : array, shape=(grid_size[1], grid_size[0]), default=None
        The energies at each grid point. Not needed for a grid being
        initialized. If None, all values are set to 0.
    resolution : int, default=1
        The number of energies to include between states on the grid.
        This serves as a type of resolution of the landscape.
    """

    def __init__(
            self, grid_size=None,  x1_coords=None, x2_coords=None, values=None,
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

    def add_barrier(self,height=3,width=3,start=(0,0),dist=5,angle=90,draw=True):
        '''Changes values in place and returns end point of barrier drawn'''
        y0 = start[0]
        x0 = start[1]
        # Avoid infinite slope
        if angle in [0,90,180,270,360]:
            angle += 0.01
        x1,y1 = draw_line(angle,dist,x0,y0)
        p1 = np.array([y0,x0])
        p2 = np.array([y1,x1])
        points = []
        #Line segment distance to all grid points - SHOULD VECTORIZE
        for i in range(self.values.shape[0]):
            for j in range(self.values.shape[1]):
                p3 = np.array([i,j])
                dist_to_line = point_seg_dist(p1,p2,p3)
                if dist_to_line < (width / 2.5):
                    points.append(p3)
                    if draw:
                        self.values[i,j] += height
        return points

    def add_noise(
            self, gaussians_per_axis=None, height_range=[-0.1, 0.1],
            width_range=[0.85, 1.15], rigidity=0):
        noise = gaussian_noise(
            self.x1_coords, self.x2_coords,
            gaussians_per_axis = gaussians_per_axis, height_range = height_range,
            width_range = width_range, rigidity = rigidity)
        return landscape(
            grid_size=self.grid_size, x1_coords=self.x1_coords,
            x2_coords=self.x2_coords, values=self.values+noise)

    def plot(
            self, title='potential energy landscape', cmap='seismic',
            **kwargs):
        plt.figure(
            title, figsize=(
                self.x1_coords.max()/self.x2_coords.max()*10, 8))
        plt.xlim((self.x1_coords[0,0], self.x1_coords[0,-1]))
        plt.ylim((self.x2_coords[0,0], self.x2_coords[-1,0]))
        plt.pcolormesh(
            self.x1_coords, self.x2_coords, self.values, cmap=cmap, **kwargs)
        plt.colorbar()
        plt.show()
        return

    def cplot(
            self, title='potential energy landscape', cmap='RdYlBu_r',
            n_bins=10, show_plot=True, grid=True, **kwargs):
        # get X, Y, and Z coords
        X = self.x1_coords
        Y = self.x2_coords
        Z = self.values
        # setup figure
        fig = plt.figure(
            title, figsize=(
                self.x1_coords.max()/self.x2_coords.max()*10, 8))
        ax = fig.add_subplot(1,1,1)
        # get appropriate levels
        try:
            levels = kwargs['levels']
            kwargs = removekey(kwargs,'levels')
            try:
                norm = kwargs['norm']
                if type(norm) is type(colors.LogNorm()):
                    tick_values = np.logspace(
                        start=np.log10(np.min(levels)),
                        stop=np.log10(np.max(levels)),
                        num=len(levels), base=10.0)
                else:
                    raise
            except:
                tick_values = np.linspace(
                    start=np.min(levels),
                    stop=np.max(levels),
                    num=len(levels))
        except:
            try:
                # test if norm exists
                norm = kwargs['norm']
                if type(norm) is type(colors.LogNorm()):
                    levels = np.logspace(
                        #Z.min(), Z.max(), nbins, endpoint=True)
                        start=np.log10(Z.min()),
                        stop=np.log10(Z.max()),
                        num=n_bins, base=10.0)
                else:
                    levels = mpl.ticker.MaxNLocator(n_bins=n_bins).tick_values(Z.min(), Z.max())
            except:
                levels = mpl.ticker.MaxNLocator(n_bins=n_bins).tick_values(Z.min(), Z.max())
            tick_values = np.linspace(Z.min(), Z.max(), len(levels))

        CS = ax.contourf(
            X, Y, Z, cmap=cmap, levels=levels, origin='lower', zorder=0, **kwargs)
        if grid:
            # set grid
            intervals = 1
            loc = plticker.MultipleLocator(base=intervals)
            ax.yaxis.set_major_locator(loc)
            ax.xaxis.set_major_locator(loc)
            plt.grid('on', linestyle='-', linewidth=1, color='black', zorder=1)
        # make colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cb = fig.colorbar(CS, cax=cbar_ax, ticks=levels)
        cb.ax.set_yticklabels([str(i) for i in tick_values])
        if show_plot:
            plt.show()
        return fig, ax

    def save_fig(self, output_name, title='potential energy landscape'):
        plt.figure(title)
        plt.xlim((self.x1_coords[0,0], self.x1_coords[0,-1]))
        plt.ylim((self.x2_coords[0,0], self.x2_coords[-1,0]))
        plt.pcolormesh(self.x1_coords, self.x2_coords, self.values)
        plt.colorbar()
        plt.savefig(output_name)
        plt.show()

    def to_probs(self):
        T = surface_to_probs(
            self.x1_coords, self.x2_coords, self.values, self.grid_size)
        return T

    def save(self, output_name, txt=False, txt_fmt='%d %d %d %f'):
        if txt:
            x1_coords_flat = self.x1_coords.flatten()
            x2_coords_flat = self.x2_coords.flatten()
            values_flat = self.values.flatten()
            states = np.arange(len(x1_coords_flat))
            output_data = np.array(
                list(
                    zip(states, x1_coords_flat, x2_coords_flat, values_flat)))
            np.savetxt(
                output_name, output_data, fmt=txt_fmt,
                header='state x1 x2 energy')
        else:
            output_dict = {
                'x1_coords' : self.x1_coords,
                'x2_coords' : self.x2_coords,
                'landscape' : self.values}
            io.saveh(output_name, **output_dict)

    def load(input_name):
        load_dict = io.loadh(input_name)
        x1_coords = load_dict['x1_coords']
        x2_coords = load_dict['x2_coords']
        values = load_dict['landscape']
        return landscape(
            x1_coords=x1_coords, x2_coords=x2_coords, values=values)

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


