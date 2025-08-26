"""
Initial conditions for a periodic version of the Richtmyer-Meshkov
instability
"""

from prep_jax import *
from modules.geometry.grid import *
from modules.numerical.computation import smoothed_jump

def periodic_richtmyer_meshkov(mesh, rho_c, p_c):
    """
    Generate a periodic adaptation of the classical Richtmyer-Meshkov experiment.
    """
    slope = 200

    x_rho_up = mesh[1] - DOMAIN_SIZE[1] * (0.7 + 0.05 * jnp.cos(2 * jnp.pi * mesh[0] / DOMAIN_SIZE[0]))
    x_rho_down = mesh[1] - DOMAIN_SIZE[1] * (0.3 + 0.05 * jnp.cos(2 * jnp.pi * mesh[0] / DOMAIN_SIZE[0]))
    x_rho_2 = jnp.abs(mesh[1] - DOMAIN_SIZE[1] * 0.5) - 0.1 * DOMAIN_SIZE[1]
    rho = smoothed_jump(x_rho_2, 0.5, 0.0, slope) + smoothed_jump(x_rho_up, 1.0, 0.0, slope) + smoothed_jump(x_rho_down, 0.0, 1.0, slope) 
    u = jnp.zeros_like(mesh[0]) 
    v = jnp.zeros_like(mesh[0])  
    p = smoothed_jump(x_rho_2, 7.5, 1.0, slope)
    return jnp.stack((rho, u, v, p), axis = 0)

#just a axially symmetric richtmyer-meshkov instability
def blast_wave_2d(mesh, rho_c, p_c):
    """
    Generate a blast wave initial condition.
    
    Returns:
    jnp.ndarray: A stack of 2D arrays representing the blast wave initial condition.
    """
    rho = rho_c * jnp.ones_like(mesh[0]) 
    u = jnp.zeros_like(mesh[0])  # Zero velocity field
    v = jnp.zeros_like(mesh[0])  # Zero velocity field

    circle = jnp.sqrt((mesh[0] - DOMAIN_SIZE[0]/2)**2 + (mesh[1] - DOMAIN_SIZE[1]/2)**2) - 0.1
    p = smoothed_jump(circle, 2.5, 0.1, 200.0) * p_c  # Pressure field with a jump
    return jnp.stack((rho, u, v, p), axis=0)  # Stack to create a 2D array with shape (4, n_x, n_y)
