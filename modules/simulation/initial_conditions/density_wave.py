"""
Density Wave initial conditions
"""

from prep_jax import *
from modules.geometry.grid import *
from modules.thermodynamics.EOS import *

rho_c, T_c, p_c = molecule.critical_point

@jax.jit
def density_wave_1d(mesh):
    """
    Generate a 1D density wave pattern.
    
    Returns:
    jnp.ndarray: A stack of 2D arrays representing the density wave initial condition.
    """

    rho = rho_c * (1 + 0.25 * jnp.sin(2 * jnp.pi * mesh[0] / DOMAIN_SIZE[0]))
    u = 10 * jnp.ones_like(mesh[0]) 
    p = 2 * p_c * jnp.ones_like(mesh[0])  
    return jnp.stack((rho, u, p), axis=0), 0 #rvp  # Stack to create a 2D array with shape (4, n_x, n_y)





# @jax.jit
# def density_wave_1d(mesh):
#     """
#     Generate a 1D density wave pattern.
    
#     Returns:
#     jnp.ndarray: A stack of 2D arrays representing the density wave initial condition.
#     """

#     k = 10

#     u = 10.0 * jnp.ones_like(mesh[0]) 
#     p = p_c * (1.5 + 1e-1 * jnp.sin(2 * jnp.pi * k * mesh[0] / DOMAIN_SIZE[0]))
#     T = T_c * (1.5 + 1e-1 * jnp.sin(2 * jnp.pi * k * mesh[0] / DOMAIN_SIZE[0]))
#     return jnp.stack((u, p, T), axis=0), 1 #vpT  # Stack to create a 2D array with shape (4, n_x, n_y)