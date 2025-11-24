"""
Density Wave initial conditions
"""

from prep_jax import *
from modules.geometry.grid import *

def density_wave_1d(mesh, molecule):
    """
    Generate a 1D density wave pattern.
    
    Returns:
    jnp.ndarray: A stack of 2D arrays representing the density wave initial condition.
    """

    rho_c, T_c, p_c = molecule.critical_points

    rho = rho_c * (1 + 0.25 * jnp.sin(2 * jnp.pi * mesh[0] / DOMAIN_SIZE[0]))
    u = 10 * jnp.ones_like(mesh[0]) 
    p = 2 * p_c * jnp.ones_like(mesh[0])  
    return jnp.stack((rho, u, p), axis=0), 'rvp'  # Stack to create a 2D array with shape (4, n_x, n_y)

def density_wave_1d_isothermal(mesh, molecule):
    """
    Generate a 1D density wave pattern.
    
    Returns:
    jnp.ndarray: A stack of 2D arrays representing the density wave initial condition.
    """

    rho_c, T_c, p_c = molecule.critical_points

    rho = rho_c * (1 + 0.25 * jnp.sin(2 * jnp.pi * mesh[0] / DOMAIN_SIZE[0]))
    u = 10 * jnp.ones_like(mesh[0]) 
    T = 2 * T_c * jnp.ones_like(mesh[0])  
    return jnp.stack((rho, u, T), axis=0), 'rvt'  # Stack to create a 2D array with shape (4, n_x, n_y)
