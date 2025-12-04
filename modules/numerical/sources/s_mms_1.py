"""
Manufactured solution #1 for the Euler equation
"""

from prep_jax import *
from config.conf_numerical import *
from modules.geometry.grid import *
from modules.thermodynamics.EOS import *
from modules.simulation.initial_conditions.u_mms_1 import solution_u_mms_1

rho_c, T_c, p_c = molecule.critical_point

def source_1d(u_num, T_num, t):
    """
    Source term s for manufactured solution number 1

    source term s enters Euler as: du/dt + df/dx = s
    """

    n_x = GRID_RESOLUTION[0]
    d_x = GRID_SPACING[0]

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

    #compute derivatives (could have been AD, but I like suffering)
    rho_t = - rho_c * omega * Arho * jnp.cos(k * mesh[0] / DOMAIN_SIZE[0]) * jnp.sin(omega * t)
    rho_x = - rho_c * k / DOMAIN_SIZE[0] *  Arho * jnp.sin(k * mesh[0] / DOMAIN_SIZE[0]) * jnp.cos(omega * t)

    vel_t = - c * omega * Avel * jnp.sin(k * mesh[0] / DOMAIN_SIZE[0]) * jnp.sin(omega * t)
    vel_x = c * k / DOMAIN_SIZE[0] * Avel * jnp.cos(k * mesh[0] / DOMAIN_SIZE[0]) * jnp.cos(omega * t)

    T_t = T_c * omega * AT * jnp.cos(k * mesh[0] / DOMAIN_SIZE[0]) * jnp.cos(omega * t)
    T_x = - T_c * k / DOMAIN_SIZE[0] * AT * jnp.sin(k * mesh[0] / DOMAIN_SIZE[0]) * jnp.sin(omega * t)

    #source rho
    s_rho = rho_t + rho_x * vel + rho * vel_x

    #source m
    p   = pressure(rho, T)
    p_t = pressure_rho(rho, T) * rho_t + pressure_T(rho, T) * T_t
    p_x = pressure_rho(rho, T) * rho_x + pressure_T(rho, T) * T_x

    s_m = rho_t * vel + rho * vel_t + rho_x * vel**2 + 2 * rho * vel * vel_x + p_x

    #source E
    e = internal_energy(rho, T)
    E = rho * e + 1/2 * rho * vel**2

    e_t = internal_energy_rho(rho, T) * rho_t + internal_energy_T(rho, T) * T_t
    E_t = rho_t * e + rho * e_t + 1/2 * rho_t * vel**2 + rho * vel * vel_t

    e_x = internal_energy_rho(rho, T) * rho_x + internal_energy_T(rho, T) * T_x
    E_x = rho_x * e + rho * e_x + 1/2 * rho_x * vel**2 + rho * vel * vel_x

    s_E = E_t + vel_x * (E + p) + vel * (E_x + p_x)

    return jnp.stack((s_rho, s_m, s_E), axis = 0)

