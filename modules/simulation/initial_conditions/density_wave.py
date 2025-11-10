"""
Density Wave initial conditions
"""

from prep_jax import *
from modules.geometry.grid import *

def density_wave_3d(mesh, molecule):
    """
    Generate a 3D density wave pattern.
    
    Returns:
    jnp.ndarray: A stack of 3D arrays representing the density wave initial condition.
    """

    rho_c, T_c, p_c = molecule.critical_points

    rho = rho_c * (2 + 0.25 * jnp.sin(2 * jnp.pi * mesh[0] / DOMAIN_SIZE[0]))
    u = jnp.ones_like(mesh[0])  # ones velocity field
    v = jnp.zeros_like(mesh[0])  # Zero velocity field
    w = jnp.zeros_like(mesh[0])  # Zero velocity field
    p = 2 * p_c * jnp.ones_like(mesh[0])  # Uniform pressure field
    return jnp.stack((rho, u, v, w, p), axis=0), 'rvp'  # Stack to create a 3D array with shape (5, n_x, n_y, n_z)

def density_wave_2d(mesh, molecule):
    """
    Generate a 2D density wave pattern.
    
    Returns:
    jnp.ndarray: A stack of 2D arrays representing the density wave initial condition.
    """

    rho_c, T_c, p_c = molecule.critical_points

    rho = rho_c * (1 + 0.25 * jnp.sin(2 * jnp.pi * mesh[0] / DOMAIN_SIZE[0]))
    u = 10 * jnp.ones_like(mesh[0]) 
    v = jnp.zeros_like(mesh[0])  
    p = 2 * p_c * jnp.ones_like(mesh[0])  
    return jnp.stack((rho, u, v, p), axis=0), 'rvp'  # Stack to create a 2D array with shape (4, n_x, n_y)

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
