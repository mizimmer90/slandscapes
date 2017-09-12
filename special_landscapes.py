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

