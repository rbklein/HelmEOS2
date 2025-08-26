"""
Taylor-Green vortex initial conditions
"""

from prep_jax import *
from modules.geometry.grid import *

def Taylor_Green_vortex_3d(mesh, rho_c, p_c):
    """
    Generate a 3D Taylor-Green vortex pattern.
    
    Returns:
    jnp.ndarray: A stack of 3D arrays representing the Taylor-Green vortex initial condition.
    """
    rho0 = 0.01 * rho_c
    U0 = 1
    M0 = 0.1
    rho = rho0 * jnp.ones_like(mesh[0])  # Uniform density field
    u = U0 * jnp.sin(2 * jnp.pi * mesh[0] / DOMAIN_SIZE[0]) * jnp.cos(2 * jnp.pi * mesh[1] / DOMAIN_SIZE[1]) * jnp.cos(2 * jnp.pi * mesh[2] / DOMAIN_SIZE[2])
    v = -U0 * jnp.cos(2 * jnp.pi * mesh[0] / DOMAIN_SIZE[0]) * jnp.sin(2 * jnp.pi * mesh[1] / DOMAIN_SIZE[1]) * jnp.cos(2 * jnp.pi * mesh[2] / DOMAIN_SIZE[2])
    w = jnp.zeros_like(mesh[0])  # Zero velocity field in the z-direction
    p =  p_c + rho0 * U0**2 /16 * (jnp.cos(4 * jnp.pi * mesh[0] / DOMAIN_SIZE[0]) + jnp.cos(4 * jnp.pi * mesh[1] / DOMAIN_SIZE[1]))*(jnp.cos(4 * jnp.pi * mesh[2] / DOMAIN_SIZE[2] + 2))
    return jnp.stack((rho, u, v, w, p), axis=0)