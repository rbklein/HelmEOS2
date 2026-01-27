"""
    Functions for calculating fluxes in numerical simulations.
"""

from prep_jax               import *
from config.conf_numerical  import *
from config.conf_geometry   import *

from modules.geometry.grid          import CELL_VOLUME
from modules.simulation.boundary    import apply_boundary_conditions, apply_temperature_boundary_condition

''' Consistency checks '''

KNOWN_FLUX_TYPES = ["NAIVE", "KEEP", "RANOCHA_IDEAL", "ISMAIL_ROE_IDEAL", "CHANDRASHEKAR_IDEAL", "KUYA", "AIELLO"]
KNOWN_DISCRETE_GRADIENTS = ["SYM_ITOH_ABE", "GONZALEZ", "NONE"] 

assert NUMERICAL_FLUX in KNOWN_FLUX_TYPES, f"Unknown numerical flux: {NUMERICAL_FLUX}"

#assert discrete gradient specification here, import in numerical flux definition files (discrete gradient implementations are flux-specific) 
assert DISCRETE_GRADIENT in KNOWN_DISCRETE_GRADIENTS, f"Unknown discrete gradient: {DISCRETE_GRADIENT}"

''' Functions for numerical fluxes '''
match NUMERICAL_FLUX:
    case "NAIVE":
        if N_DIMENSIONS == 1:
            raise NotImplementedError(f"Naive flux not implemented in 1D")
            #from modules.numerical.fluxes.naive import div_naive_1d as flux_div
        elif N_DIMENSIONS == 2:
            from modules.numerical.fluxes.naive import div_naive_2d as flux_div
        elif N_DIMENSIONS == 3:
            from modules.numerical.fluxes.naive import div_naive_3d as flux_div
    case "KEEP":  
        if N_DIMENSIONS == 1:
            from modules.numerical.fluxes.keep_dg import div_keep_dg_1d as flux_div  
        elif N_DIMENSIONS == 2:
            from modules.numerical.fluxes.keep_dg import div_keep_dg_2d as flux_div
        elif N_DIMENSIONS == 3:
            from modules.numerical.fluxes.keep_dg import div_keep_dg_3d as flux_div
    case "RANOCHA_IDEAL":
        if N_DIMENSIONS == 1:
            from modules.numerical.fluxes.ranocha import div_ranocha_1d as flux_div
        elif N_DIMENSIONS == 2:
            raise NotImplementedError(f"Ranocha ideal flux not implemented in 2D")
        elif N_DIMENSIONS == 3:
            raise NotImplementedError(f"Ranocha ideal flux not implemented in 3D")
    case "ISMAIL_ROE_IDEAL":
        if N_DIMENSIONS == 1:
            from modules.numerical.fluxes.ismail_roe import div_ismail_roe_1d as flux_div
        elif N_DIMENSIONS == 2:
            raise NotImplementedError(f"Ismail and Roe ideal flux not implemented in 2D")
        elif N_DIMENSIONS == 3:
            raise NotImplementedError(f"Ismail and Roe ideal flux not implemented in 3D")
    case "CHANDRASHEKAR_IDEAL":
        if N_DIMENSIONS == 1:
            from modules.numerical.fluxes.chandrashekar import div_chandrashekar_1d as flux_div
        elif N_DIMENSIONS == 2:
            raise NotImplementedError(f"Chandrashekar ideal flux not implemented in 2D")
        elif N_DIMENSIONS == 3:
            raise NotImplementedError(f"Chandrashekar ideal flux not implemented in 3D")
    case "KUYA":
        if N_DIMENSIONS == 1:
            from modules.numerical.fluxes.kuya import div_kuya_1d as flux_div
        elif N_DIMENSIONS == 2:
            from modules.numerical.fluxes.kuya import div_kuya_2d as flux_div
        elif N_DIMENSIONS == 3:
            from modules.numerical.fluxes.kuya import div_kuya_3d as flux_div
    case "AIELLO":
        if N_DIMENSIONS == 1:
            raise NotImplementedError(f"Chandrashekar ideal flux not implemented in 1D")
        elif N_DIMENSIONS == 2:
            raise NotImplementedError(f"Chandrashekar ideal flux not implemented in 2D")
        elif N_DIMENSIONS == 3:
            from modules.numerical.fluxes.aiello import div_aiello_3d as flux_div
    case _:
        raise ValueError(f"Unknown numerical flux: {NUMERICAL_FLUX}")

def dudt(u, T):
    """
    Calculate the time derivative of u using the specified numerical flux.

    Parameters:
    u (array-like): Current state of the system.
    T (array-like): Current temperature of the system

    Returns:
    array-like: Time derivative of u.
    """
    u = apply_boundary_conditions(u, T)  # Apply boundary conditions to u
    T = apply_temperature_boundary_condition(u, T)
    return - flux_div(u, T) / CELL_VOLUME