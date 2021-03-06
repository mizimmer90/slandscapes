import numpy as np
from . import landscape


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


def slant_landscape(
        grid_size, gradient=1.0, resolution=1):
    l = landscape(grid_size=grid_size, resolution=resolution)
    l.values = np.array(
        [np.arange(l.values.shape[1]) * gradient] * l.values.shape[0])
    return l

def single_barrier(grid_x=40,grid_y=20,barrier_height=3,add_noise=False):
    l = landscape((grid_x,grid_y))
    x_col = int(grid_x / 2)
    l.values[:,x_col] = barrier_height
    if add_noise:
        l = l.add_noise(gaussians_per_axis=10, height_range=[-(barrier_height/2),(barrier_height/2)])
    return l

def multiple_paths(grid_x=40,path_width=5,barrier_height=3,add_noise=False,number_paths=3):
    grid_y = (path_width * number_paths) + (number_paths - 1)
    l = landscape((grid_x,grid_y))
    barrier_list = [path_width]
    x_col = int(grid_x/5)
    for i in range(2,number_paths):
        barrier_list.append((path_width*i) + (i-1))
    l.values[barrier_list,x_col:] = barrier_height
    if add_noise:
        l = l.add_noise(height_range=[-(barrier_height*(1/2)),(barrier_height*(1/2))])
    return l

def multiple_barriers(well_width=100,well_depth=50,barrier_height=3,add_noise=False,number_barriers=3):
    barrier_blocks = ((number_barriers * 2) + 1)
    grid_x = int(well_depth * barrier_blocks)
    grid_y = int(well_width * (5/4))
    l = landscape((grid_x,grid_y))
    j = 1
    for i in range(2,barrier_blocks,2):
        if (j % 2 == 1):
            l.values[-1,((i-1)*well_depth):((i-1)*well_depth + well_depth -1)] = barrier_height
            l.values[int((well_width/4)),((i-1)*well_depth):((i-1)*well_depth + well_depth -1)] = barrier_height
            l.values[int((well_width/4)):-1,((i-1)*well_depth + well_depth -1)] =  barrier_height
        else:
            l.values[0,((i-1)*well_depth):((i-1)*well_depth + well_depth -1)] = barrier_height
            l.values[well_width-1,((i-1)*well_depth):((i-1)*well_depth + well_depth -1)] = barrier_height
            l.values[0:well_width,((i-1)*well_depth + well_depth -1)] =  barrier_height
        j = j+1
    if add_noise:
        l = l.add_noise(height_range=[-(barrier_height*(1/2)),(barrier_height*(1/2))])
    return l
