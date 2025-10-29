"""
    Functions for calculating fluxes in numerical simulations.
"""

from prep_jax import *
from config.conf_numerical import *
from config.conf_geometry import *
from modules.simulation.boundary import apply_boundary_conditions


''' Consistency checks '''

KNOWN_FLUX_TYPES = ["NAIVE", "KEEP"]
KNOWN_DISCRETE_GRADIENTS = ["SYM_ITOH_ABE", "GONZALEZ", "NONE"] 

assert NUMERICAL_FLUX in KNOWN_FLUX_TYPES, f"Unknown numerical flux: {NUMERICAL_FLUX}"

#assert discrete gradient specification here, import in numerical flux definition files (discrete gradient implementations are flux-specific) 
assert DISCRETE_GRADIENT in KNOWN_DISCRETE_GRADIENTS, f"Unknown discrete gradient: {DISCRETE_GRADIENT}"

''' Functions for numerical fluxes '''
from modules.geometry.grid import CELL_VOLUME

match NUMERICAL_FLUX:
    case "NAIVE":
        if N_DIMENSIONS == 2:
            from modules.numerical.fluxes.naive import div_naive_2d as flux_div
        elif N_DIMENSIONS == 3:
            from modules.numerical.fluxes.naive import div_naive_3d as flux_div
    case "KEEP":    
        if N_DIMENSIONS == 2:
            from modules.numerical.fluxes.keep_dg import div_keep_dg_2d as flux_div
        elif N_DIMENSIONS == 3:
            from modules.numerical.fluxes.keep_dg import div_keep_dg_3d as flux_div
    case _:
        raise ValueError(f"Unknown numerical flux: {NUMERICAL_FLUX}")


def dudt(u):
    """
    Calculate the time derivative of u using the specified numerical flux.

    Parameters:
    u (array-like): Current state of the system.

    Returns:
    array-like: Time derivative of u.
    """
    u = apply_boundary_conditions(u)  # Apply boundary conditions to u
    return - flux_div(u) / CELL_VOLUME