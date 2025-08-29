"""
    This file will contain functionality that constructs a sequence of post processing steps from available functions (defined in plot.py, etc.) 
    as requested by the user. The sequence of functions will be carried out by a single function called post, that takes a sequence of input parameters and the data u.
"""

from prep_jax import *
from config.conf_postprocess import *
from config.conf_geometry import *
from modules.thermodynamics.EOS import *
from modules.numerical.computation import spatial_average

import matplotlib.pyplot as plt
from matplotlib import colormaps

''' Consistency checks '''

KNOWN_POSTPROCESSING_STEPS = [
    "DENSITY",
    "MOMENTUM",
    "TOTAL_ENERGY",
    "PRESSURE",
    "PRESSURE_FLUCTUATIONS",
    "TEMPERATURE",
    "ENTROPY",
    "VELOCITY",
    "VORTICITY",
    "ENERGY_SPECTRUM",
    "PV",
    "CRITICAL_DISTANCE",
    "TOTAL_ENTROPY",
    "SPEED_OF_SOUND",
    "LOCAL_MACH",
]

SCALAR_FIELDS = [
    "DENSITY",
    "MOMENTUM",
    "TOTAL_ENERGY",
    "PRESSURE",
    "PRESSURE_FLUCTUATIONS",
    "TEMPERATURE",
    "ENTROPY",
    "VELOCITY",
    "VORTICITY",
    "CRITICAL_DISTANCE",
    "SPEED_OF_SOUND",
    "LOCAL_MACH",
]

INTEGRAL_VALUES = [
    "TOTAL_ENTROPY",
]

for step in PLOT_SEQUENCE:
    assert step in KNOWN_POSTPROCESSING_STEPS, f"Unknown post-processing step: {step}"

if set(PLOT_SEQUENCE) & set(SCALAR_FIELDS):
    if N_DIMENSIONS == 2:
        from modules.postprocess.plot.scalar_field import plot_scalar_field_2d as plot_scalar_field
        from modules.postprocess.plot.scalar_field import update_scalar_field_2d as update_scalar_field
    elif N_DIMENSIONS == 3:
        from modules.postprocess.plot.scalar_field import plot_scalar_field_3d as plot_scalar_field
        from modules.postprocess.plot.scalar_field import update_scalar_field_3d as update_scalar_field

if set(PLOT_SEQUENCE) & set(INTEGRAL_VALUES):
    from modules.postprocess.plot.spatial_integral import plot_integral, update_integral
    from modules.geometry.grid import CELL_VOLUME

if "VORTICITY" in PLOT_SEQUENCE:
    if N_DIMENSIONS == 2:
        from modules.postprocess.derived_quantities.vorticity import vorticity_2d as vorticity
    elif N_DIMENSIONS == 3:
        from modules.postprocess.derived_quantities.vorticity import vorticity_3d as vorticity

if "PV" in PLOT_SEQUENCE:
    from modules.postprocess.plot.pv_diagram import plot_pv as pv_diagram

assert COLORMAP in list(colormaps), f"Unknown colormap: {COLORMAP}, use valid matplotlib cmap"

# Check if configurations are set and otherwise set them
if "MAX_TIME_SERIES_LENGTH" in locals():
    MAX_TIME_SERIES_LENGTH = locals()["MAX_TIME_SERIES_LENGTH"]
    assert MAX_TIME_SERIES_LENGTH > 0 and isinstance(MAX_TIME_SERIES_LENGTH, int), f"Maximal time series length: {MAX_TIME_SERIES_LENGTH} has to be a positive integer"
else:
    MAX_TIME_SERIES_LENGTH = -1

assert NUM_ITS_PER_UPDATE > 0 and isinstance(NUM_ITS_PER_UPDATE, int), f"Iterations per postprocessing update: {NUM_ITS_PER_UPDATE} has to be a positive integer"

''' Derived parameters '''

NUM_PLOTS = len(PLOT_SEQUENCE)
N_ROWS_FULL = NUM_PLOTS // MAX_PLOTS_PER_ROW 
N_EXTRA_ROW = NUM_PLOTS % MAX_PLOTS_PER_ROW
N_ROWS = N_ROWS_FULL + (1 if N_EXTRA_ROW > 0 else 0)

''' Set up parameters of a master grid to contain multiple centered plots '''

COLS_MASTER = MAX_PLOTS_PER_ROW
ROWS_MASTER = N_ROWS
SHAPE_MASTER = (ROWS_MASTER, COLS_MASTER)

def get_master_row_axes(row_ind, len_row):
    """
        place a row with len_row axes in the master figure
    """
    axes = []
    for m in range(len_row):
        ax = plt.subplot2grid(SHAPE_MASTER, (row_ind, m))
        axes.append(ax)
    return axes


''' Functions for post-processing '''

# very natural python code to obtain a slicing operator that slices an ndarray of shape GRID_RESOLUTION 
# into 10 parts and that handles small grid_resolutions 
slices_pv = tuple(((slice(None, None, n//10) if n//10 != 0 else slice(0, None, None)) for n in GRID_RESOLUTION))

def init_postprocess():
    """
    Initialize the post-processing sequence.
    """
    plt.ion()
    fig = plt.figure(figsize = (5 * MAX_PLOTS_PER_ROW, 5 * N_ROWS))
    rows = []
    for i in range(N_ROWS_FULL):
        rows.append(get_master_row_axes(i, MAX_PLOTS_PER_ROW))
    if N_EXTRA_ROW > 0:
        rows.append(get_master_row_axes(N_ROWS - 1, N_EXTRA_ROW))

    return fig, rows


#IMPROVEMENT: CONSTRUCT UPDATE FUNCTION BY PUTTING THE NECESSARY 
# PLOTTING FUNCTIONS IN THE PLOT_GRID LIST AS WELL
def plot_postprocess(u, fig, rows, cmap = 'viridis'):
    """
    Plot the post-processing sequence
    """
    rho = u[0]
    T = temperature(u)
    p = pressure(rho, T)

    plot_grid = []

    count = 0
    for row in rows:
        plot_row = []
        for ax in row:
            step = PLOT_SEQUENCE[count]
            if step in SCALAR_FIELDS:
                match step:
                    case "DENSITY":
                        field = rho 
                        title = "Density"
                    case "MOMENTUM":
                        field = jnp.linalg.norm(u[1:1+N_DIMENSIONS], axis = 0)
                        title = "Momentum Magnitude"
                    case "TOTAL_ENERGY":
                        field = u[-1]
                        title = "Total Energy"
                    case "PRESSURE":
                        field = p
                        title = "Pressure"
                    case "PRESSURE_FLUCTUATIONS":
                        avg_p = spatial_average(p)
                        field = (p - avg_p) / avg_p
                        title = "Pressure fluctuations"
                    case "TEMPERATURE":
                        field = T
                        title = "Temperature"
                    case "ENTROPY":
                        field = entropy(rho, T)
                        title = "Entropy"
                    case "VELOCITY":
                        field = jnp.linalg.norm(u[1:1+N_DIMENSIONS] / rho, axis = 0)
                        title = "Velocity Magnitude"
                    case "VORTICITY":
                        if N_DIMENSIONS == 2:
                            field = vorticity(u)
                            title = "Vorticity"
                        elif N_DIMENSIONS == 3:
                            #do something else when 3D plotting is implemented
                            field = jnp.linalg.norm(vorticity(u), axis=0)
                            title = "Vorticity Magnitude"
                    case "CRITICAL_DISTANCE":
                        field = jnp.sqrt((rho / rho_c - 1.0)**2 + (T / T_c - 1.0)**2 + (p / p_c - 1.0)**2)
                        title = "Critical distance"
                    case"SPEED_OF_SOUND":
                        field = speed_of_sound(rho, T)
                        title = "Speed of Sound"
                    case "LOCAL_MACH":
                        field = jnp.linalg.norm(u[1:N_DIMENSIONS] / rho, axis = 0) / speed_of_sound(rho, T)
                        title = "Mach number"
                    case _:
                        raise ValueError(f"Postprocessing for scalar field: {step}, not implemented")

                #field = np.array(jax.device_put(field, cpus[0]))
                im, cbar = plot_scalar_field(field, fig, ax, title = title, cmap = cmap)
                plot_row.append((im, cbar))

            elif step in INTEGRAL_VALUES:
                match step:
                    case "TOTAL_ENTROPY":
                        integral = jnp.sum(rho * entropy(rho, T)) * CELL_VOLUME
                        title = "Total Entropy"
                
                line, ax_new = plot_integral(integral, fig, ax, title)
                plot_row.append((line, ax_new))

            else:
                match step:
                    case "PV":
                        rho_arr = jnp.ravel(rho[slices_pv])
                        p_arr = jnp.ravel(p[slices_pv])
                        ax = pv_diagram(rho_arr, p_arr, fig, ax)
                        plot_row.append((ax))
                    case "ENERGY_SPECTRUM":
                        plot_row.append((ax))
                        pass  # Placeholder for energy spectrum plot
                    case _:
                        raise ValueError(f"Postprocessing for option: {step}, not implemented")

            count += 1
        plot_grid.append(plot_row)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.tight_layout()
    plt.pause(0.0001)
    
    return plot_grid


def update_postprocess(u, fig, plot_grid):
    """
    Update the post-processing sequence
    """
    rho = u[0]
    T = temperature(u)
    p = pressure(rho, T)

    count = 0
    for plot_row in plot_grid:
        for plot_vars in plot_row:
            step = PLOT_SEQUENCE[count]
            if step in SCALAR_FIELDS:
                match step:
                    case "DENSITY":
                        field = rho 
                    case "MOMENTUM":
                        field = jnp.linalg.norm(u[1:1+N_DIMENSIONS], axis = 0)
                    case "TOTAL_ENERGY":
                        field = u[-1]
                    case "PRESSURE":
                        field = p
                    case "PRESSURE_FLUCTUATIONS":
                        avg_p = spatial_average(p)
                        field = (p - avg_p) / avg_p
                    case "TEMPERATURE":
                        field = T
                    case "ENTROPY":
                        field = entropy(rho, T)
                    case "VELOCITY":
                        field = jnp.linalg.norm(u[1:1+N_DIMENSIONS] / rho, axis = 0)
                    case "VORTICITY":
                        if N_DIMENSIONS == 2:
                            field = vorticity(u)
                        elif N_DIMENSIONS == 3:
                            #do something else when 3D plotting is implemented
                            field = jnp.linalg.norm(vorticity(u), axis=0)
                    case "CRITICAL_DISTANCE":
                        field = jnp.sqrt((rho / rho_c - 1.0)**2 + (T / T_c - 1.0)**2 + (p / p_c - 1.0)**2)
                    case"SPEED_OF_SOUND":
                        field = speed_of_sound(rho, T)
                    case "LOCAL_MACH":
                        field = jnp.linalg.norm(u[1:N_DIMENSIONS] / rho, axis = 0) / speed_of_sound(rho, T)
                    case _:
                        raise ValueError(f"Postprocessing for scalar field: {step}, not implemented")

                #field = np.array(jax.device_put(field, cpus[0]))
                im, cbar = plot_vars
                update_scalar_field(field, im, cbar)

            elif step in INTEGRAL_VALUES:
                match step:
                    case "TOTAL_ENTROPY":
                        integral = jnp.sum(rho * entropy(rho, T)) * CELL_VOLUME
                
                line, ax = plot_vars
                update_integral(integral, line, ax, nits = NUM_ITS_PER_UPDATE, max_length=MAX_TIME_SERIES_LENGTH)

            else:
                match step:
                    case "PV":
                        ax = plot_vars
                        rho_arr = jnp.ravel(rho[slices_pv])
                        p_arr = jnp.ravel(pressure(rho, T)[slices_pv])
                        pv_diagram(rho_arr, p_arr, fig, ax)
                    case "ENERGY_SPECTRUM":
                        pass  # Placeholder for energy spectrum plot
                    case _:
                        raise ValueError(f"Postprocessing for option: {step}, not implemented")

            count += 1

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.0001)

