"""
    Contains functions that construct the heat flux divergence
"""

from prep_jax import *
from config.conf_numerical import *
from config.conf_geometry import *
from modules.simulation.boundary import apply_boundary_conditions

''' Consistency checks '''

KNOWN_HEAT_FLUX_TYPES = ["NAIVE"]

assert NUMERICAL_HEAT_FLUX in KNOWN_HEAT_FLUX_TYPES, f"Unknown numerical heat flux: {NUMERICAL_HEAT_FLUX}"

''' Functions for heat fluxes '''
from modules.geometry.grid import CELL_VOLUME

match NUMERICAL_HEAT_FLUX:
    case "NAIVE":
        if N_DIMENSIONS == 2:
            from modules.numerical.heat_fluxes.naive_heat_flux import div_naive_heat_flux_2d as heat_flux_div
        elif N_DIMENSIONS == 3:
            from modules.numerical.heat_fluxes.naive_heat_flux import div_naive_heat_flux_3d as heat_flux_div
    case _:
        raise ValueError(f"Unknown numerical heat flux: {NUMERICAL_HEAT_FLUX}")
    
def dudt(u):
    """
    Calculate the time derivative of u using the specified numerical heat flux.

    Parameters:
    u (array-like): Current state of the system.

    Returns:
    array-like: Time derivative of u.
    """
    u = apply_boundary_conditions(u)
    return heat_flux_div(u) / CELL_VOLUME