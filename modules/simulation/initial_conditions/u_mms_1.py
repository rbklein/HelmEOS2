"""
Initial condition manufactured solution 1
"""

from prep_jax import *
from modules.geometry.grid import *
from modules.thermodynamics.EOS import *

rho_c, T_c, p_c = molecule.critical_point

@jax.jit
def solution_u_mms_1(mesh, t):
    #set parameters
    k = omega = 2 * jnp.pi
    rho0, Arho = 1.2, 0.01
    T0,   AT   = 1.2, 0.01

    c = speed_of_sound(rho0 * rho_c * jnp.ones_like(mesh[0]), T0 * T_c * jnp.ones_like(mesh[0]))[0]
    vel0, Avel = 0.1, 0.01

    #compute solutions
    rho = rho_c * (rho0 + Arho * jnp.cos(k * mesh[0] / DOMAIN_SIZE[0]) * jnp.cos(omega * t))
    vel = c *     (vel0 + Avel * jnp.sin(k * mesh[0] / DOMAIN_SIZE[0]) * jnp.cos(omega * t))
    T   = T_c   * (T0   + AT   * jnp.cos(k * mesh[0] / DOMAIN_SIZE[0]) * jnp.sin(omega * t))
    return jnp.stack((rho, vel, T), axis = 0), 2


@jax.jit
def u_mms_1(mesh):
    #set parameters
    k = omega = 2 * jnp.pi
    rho0, Arho = 1.2, 0.01
    T0,   AT   = 1.2, 0.01

    c = speed_of_sound(rho0 * rho_c * jnp.ones_like(mesh[0]), T0 * T_c * jnp.ones_like(mesh[0]))[0]
    vel0, Avel = 0.1, 0.01

    #compute initial condition
    rho = rho_c * (rho0 + Arho * jnp.cos(k * mesh[0] / DOMAIN_SIZE[0]))
    vel = c *     (vel0 + Avel * jnp.sin(k * mesh[0] / DOMAIN_SIZE[0]))
    T   = T_c   * (T0 * jnp.ones_like(mesh[0]))

    return jnp.stack((rho, vel, T), axis = 0), 2 #rvt