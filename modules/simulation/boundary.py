"""
    Functions for handling boundary conditions in simulations.
"""
import prep_jax

from config.conf_geometry import *
from config.conf_simulation import *
from modules.geometry.grid import *

''' Consistency checks '''

KNOWN_BC_TYPES = ["PERIODIC"]

assert len(BC_TYPES) == N_DIMENSIONS, "Number of boundary conditions must match number of dimensions"
for bc in BC_TYPES:
    assert isinstance(bc, tuple) and len(bc) == 2, "Each boundary condition must be a tuple of (side_0, side_1)"
    assert bc[0] in KNOWN_BC_TYPES, f"Unknown boundary condition type: {bc[0]}"
    assert (bc[0] == "PERIODIC" and bc[1] == "PERIODIC") or (bc[0] != "PERIODIC" and bc[1] != "PERIODIC") , "Periodic boundary conditions must have both sides set to 'PERIODIC'"

''' Functions for boundary conditions '''

#@partial(jax.jit, static_argnames = ("axis",))
def pad_periodic(u, axis):
    """
    Apply periodic boundary conditions by padding the array.
    
    Parameters:
    u (jnp.ndarray): The array to apply periodic boundary conditions to.
    axis (int): The axis along which to apply the periodic boundary condition.
    
    Returns:
    jnp.ndarray: The padded array with periodic boundary conditions applied.
    """
    padding = tuple((1, 1) if i == (axis + 1) else (0, 0) for i in range(N_DIMENSIONS + 1))
    return jnp.pad(u, padding, mode = 'wrap') 

#@jax.jit
def apply_boundary_conditions(u):
    # BC_TYPES is a Python tuple/list of length n_dimensions, e.g. [("PERIODIC","PERIODIC"), ...]
    for dim in range(N_DIMENSIONS):
        # check if this dimension is flagged as periodic
        if BC_TYPES[dim] == ("PERIODIC", "PERIODIC"):
            u = pad_periodic(u, axis=dim)
    return u


if __name__ == "__main__":

    # Create a sample array to test the periodic padding
    sample_array = jnp.arange(2*3*3*3).reshape((2, 3, 3, 3))  # Example 4D array (2x3x3x3)
    

    # Apply periodic padding along the first axis (0)
    padded_array = pad_periodic(sample_array, axis=2)


    padded_array = apply_boundary_conditions(sample_array)  


    print("Original Array:\n", sample_array)
    print("Padded Array:\n", padded_array)