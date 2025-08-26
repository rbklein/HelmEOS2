"""
    Function to plot a scalar field in 2D or 3D.
"""

from prep_jax import *
from config.conf_geometry import *

Lx, Ly = DOMAIN_SIZE

import matplotlib.ticker as tkr

def plot_scalar_field_2d(scalar_field, fig, ax, title ="Scalar Field", cmap ='viridis'):
    """
    Plot a scalar field in 2D or 3D.

    """
    im = ax.imshow(scalar_field.T, extent = (0, Lx, 0, Ly), origin='lower', cmap=cmap)
    cbar = fig.colorbar(im, ax = ax, label = title, format = tkr.FormatStrFormatter('%.2g'))
    return im, cbar

def update_scalar_field_2d(scalar_field, im, cbar):
    """
    Update the scalar field plot with new data.

    """
    im.set_data(scalar_field.T)
    im.set_clim(vmin=scalar_field.min(), vmax=scalar_field.max())
    cbar.update_normal(im)
    return im, cbar

def plot_scalar_field_3d(scalar_field, fig, ax, ind = None, title ="Scalar Field", cmap ='viridis'):
    pass

def update_scalar_field_3d(scalar_field, im, cbar):
    pass