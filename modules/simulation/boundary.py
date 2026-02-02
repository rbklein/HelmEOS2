"""
    Functions for handling boundary conditions in simulations.
"""
from prep_jax import *
from config.conf_geometry   import *
from config.conf_simulation import *

from jax.numpy import pad

''' Consistency checks '''

KNOWN_BC_TYPES = ["PERIODIC"]

assert len(BC_TYPES) == N_DIMENSIONS, "Number of boundary conditions must match number of dimensions"
for bc in BC_TYPES:
    assert isinstance(bc, tuple) and len(bc) == 2, "Each boundary condition must be a tuple of (side_0, side_1)"
    assert bc[0] in KNOWN_BC_TYPES, f"Unknown boundary condition type: {bc[0]}"
    assert (bc[0] == "PERIODIC" and bc[1] == "PERIODIC") or (bc[0] != "PERIODIC" and bc[1] != "PERIODIC") , "Periodic boundary conditions must have both sides set to 'PERIODIC'"

''' Functions for boundary conditions '''

def pad_periodic(u, axis):
    """
    Apply periodic boundary conditions by padding the array.
    
    Parameters:
    u (jnp.ndarray): The array containing the state to apply periodic boundary conditions to.
    axis (int): The axis along which to apply the periodic boundary condition.
    
    Returns:
    jnp.ndarray: The padded array with periodic boundary conditions applied.
    """
    padding = tuple((1, 1) if i == (axis + 1) else (0, 0) for i in range(N_DIMENSIONS + 1))
    return pad(u, padding, mode = 'wrap') 

def apply_boundary_conditions(u, T):
    # BC_TYPES is a Python tuple/list of length n_dimensions, e.g. [("PERIODIC","PERIODIC"), ...]
    for dim in range(N_DIMENSIONS):
        # check if this dimension is flagged as periodic
        if BC_TYPES[dim] == ("PERIODIC", "PERIODIC"):
            u = pad_periodic(u, axis=dim)
    return u


''' Functions for temperature boundary conditions '''


def pad_periodic_scalar(T, axis):
    """
    Apply periodic boundary conditions by padding the array.
    
    Parameters:
    T (jnp.ndarray): The array to apply periodic boundary conditions to.
    axis (int): The axis along which to apply the periodic boundary condition.
    
    Returns:
    jnp.ndarray: The padded array with periodic boundary conditions applied.
    """
    padding = tuple((1, 1) if i == axis else (0, 0) for i in range(N_DIMENSIONS))
    return pad(T, padding, mode = 'wrap') 


def apply_temperature_boundary_condition(u, T):
    for dim in range(N_DIMENSIONS):
        if BC_TYPES[dim] == ("PERIODIC", "PERIODIC"):
            T = pad_periodic_scalar(T, axis=dim) # dim-1 since T is scalar field
    return T