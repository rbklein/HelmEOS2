"""
    Function to plot a scalar field in 2D or 3D.
"""

from prep_jax import *
from config.conf_geometry import *
from config.conf_postprocess import *
from modules.geometry.grid import mesh

if N_DIMENSIONS == 2:
    Lx, Ly = DOMAIN_SIZE
elif N_DIMENSIONS == 3:
    Lx, Ly, Lz = DOMAIN_SIZE

import matplotlib.ticker as tkr

def plot_scalar_field_1d(scalar_field, fig, ax, title ="Scalar Field", cmap ='viridis'):
    """
    Plot a scalar field in 1D.
    """
    im,      = ax.plot(mesh[0], scalar_field)
    ax.set_ylabel(title)
    ax.set_xlabel('x')
    ax.grid()
    return im, ax

def update_scalar_field_1d(scalar_field, im, ax):
    """
    Update the scalar field plot with new data
    """
    im.set_ydata(scalar_field)
    ax.relim()          
    ax.autoscale_view()
    return im, ax

def plot_scalar_field_2d(scalar_field, fig, ax, title ="Scalar Field", cmap ='viridis'):
    """
    Plot a scalar field in 2D.
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

if N_DIMENSIONS == 3:
    slice_arr = []
    for i in range(3):
        if isinstance(SLICE_3D[i], slice):
            slice_arr.append(DOMAIN_SIZE[i])
    
    lengths_slice = (slice_arr[0], slice_arr[1])

def plot_scalar_field_3d(scalar_field, fig, ax, ind = None, title ="Scalar Field", cmap ='viridis'):
    """
    Plot a slice of a 3D scalar field
    """
    im = ax.imshow(scalar_field[SLICE_3D].T, extent = (0, lengths_slice[0], 0, lengths_slice[1]), origin = 'lower', cmap = cmap)
    cbar = fig.colorbar(im, ax = ax, label = title, format = tkr.FormatStrFormatter('%.2g'))
    return im, cbar

def update_scalar_field_3d(scalar_field, im, cbar):
    """
    Update the scalar field plot with new data.
    """
    im.set_data(scalar_field[SLICE_3D].T)
    im.set_clim(vmin=scalar_field.min(), vmax=scalar_field.max())
    cbar.update_normal(im)
    return im, cbar