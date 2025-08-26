"""
    Functions to plot a spatial integral evolving in time
"""

from prep_jax import *
from config.conf_geometry import *

def plot_integral(value, fig, ax, title = "Spatial Integral"):
    """
    Plot values of a spatial integral (or other scalar value) evolving in time
    """
    line, = ax.plot(value)
    ax.set_ylabel(title)
    ax.set_xlabel('n')
    ax.grid()
    return line, ax

def update_integral(value, line, ax, nits = 1, max_length = -1):
    """
    Update values in plot
    """
    t_data, integral_data = line.get_data()
    t_data          = jnp.append(t_data, t_data[-1] + nits)
    integral_data   = jnp.append(integral_data, value)

    #reduce length of series if more than max_length
    if max_length > 0 and integral_data.shape[0] > max_length:
        t_data = t_data[(t_data.shape[0] - max_length):]
        integral_data = integral_data[(integral_data.shape[0] - max_length):]
    
    line.set_data(t_data, integral_data)

    ax.relim()
    ax.autoscale_view()

    return line, ax
