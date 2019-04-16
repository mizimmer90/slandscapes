from slandscapes import landscape, rankings, mc_sampling, scalings, mc_analysis
from slandscapes.special_landscapes import multiple_barriers
import numpy as np
from enspara.msm import MSM, builders

import matplotlib.pyplot as plt
import matplotlib as mpl

def vals_for_plot(barrier_locations,counts_grid,max_counts=None):
    """Get RGB values to plot on a grid with barriers in black
    Keywords:
        barrier_locations- list of x,y tuples where the barriers are
        counts_grid- grid with counts for each point
        max_counts- int - if you want to scale to other plot
    Returns:
        outer: RGB for every grid point in proper format to plot using plot_pretty"""
    if not max_counts:
        max_counts = np.max(counts_grid)
    outer = []
    inner = []
    for r_val in range(counts_grid.shape[0]-1):
        for c_val in range(counts_grid.shape[1]):
            if (r_val,c_val) in barrier_locations:
                point = [0,0,0]
            else:
                count = counts_grid[r_val,c_val]
                rel_counts = 1 - (count / max_counts)
                point = [1,rel_counts,rel_counts]
            inner.append(point)
        outer.append(inner)
        inner = []
    return outer

#Couldn't figure out how to save this. Had to "save image as.." from jupyter notebook
def plot_cbar(max_counts):
    """Color bar gradient from white to red from 0 to max counts """
    fig, ax = plt.subplots(figsize=(0.35, 6))
    fig.subplots_adjust(bottom=0.5)

    cmap = clr.LinearSegmentedColormap.from_list('custom  red', [(1,1,1),(1,0,0)], N=256)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_counts)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('Counts')

def plot_pretty(vals,l,save_file=None):
    """Barriers in black and counts in white to red
    Keywords:
        vals - value return from vals_for_plot, 3D array size (width,length,3 [RGB])
        l - landscape object
        """
    fig = plt.figure(figsize=(l.x1_coords.max()/l.x2_coords.max()*10, 8))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)

    frame = ax.imshow(vals,origin='lower')
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    if save_file:
        fig.savefig(save_file)

#Example in Jupyter:
#    l = multiple_barriers(barrier_height=20,well_width=24,well_depth=14)
#    bar_loc = list(zip(*np.where(l.values == barrier_height)))
#    T = l.to_probs()
#    assignments = mc_sampling.adaptive_sampling(T,initial_state=1372,n_runs=1,n_clones=1,n_steps=10000, n_procs=1, n_reps=1)
    ##Filling the grid with counts
#    for val in assignments.flatten():
#        x = int(np.floor(val /98))
#        y = val % 98
#        l.values[x,y] += 1
#    l.plot()
    ##repeat for FAST, and FAST w/ PageRank, then do this
#    max_counts = np.max([np.max(l.values),np.max(f.values),np.max(p.values)])
#    c = vals_for_plot(bar_loc,l.values,max_counts=max_counts)
#    plot_pretty(c,l,save_file="../figures/conventional_1trial.png")
#    plot_cbar(max_counts=max_counts)
