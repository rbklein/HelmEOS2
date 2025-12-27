"""
    Contains functions that construct the viscous flux divergence
"""

from prep_jax               import *
from config.conf_numerical  import *
from config.conf_geometry   import *

from modules.simulation.boundary    import apply_boundary_conditions, apply_temperature_boundary_condition
from jax                            import jit

''' Consistency checks '''

KNOWN_VISCOUS_FLUX_TYPES = ["NAIVE"]

assert NUMERICAL_VISCOUS_FLUX in KNOWN_VISCOUS_FLUX_TYPES, f"Unkown numerical viscous flux: {NUMERICAL_VISCOUS_FLUX}"

''' Functions for viscous fluxes '''
from modules.geometry.grid import CELL_VOLUME

match NUMERICAL_VISCOUS_FLUX:
    case "NAIVE":
        if N_DIMENSIONS == 1:
            from modules.numerical.viscous_fluxes.naive_stress import div_naive_stress_1d as viscous_flux_div
        elif N_DIMENSIONS == 2:
            from modules.numerical.viscous_fluxes.naive_stress import div_naive_stress_2d as viscous_flux_div
        elif N_DIMENSIONS == 3:
            from modules.numerical.viscous_fluxes.naive_stress import div_naive_stress_3d as viscous_flux_div
    case _:
        raise ValueError(f"Unknown viscous numerical flux: {NUMERICAL_VISCOUS_FLUX}")
    
@jit
def dudt(u, T):
    """
    Calculate the time derivative of u using the specified numerical viscous flux.

    Parameters:
    u (array-like): Current state of the system.

    Returns:
    array-like: Time derivative of u.
    """
    u = apply_boundary_conditions(u, T)
    T = apply_temperature_boundary_condition(u, T)
    return viscous_flux_div(u, T) / CELL_VOLUME
