"""
Initial condition manufactured solution 1
"""

from prep_jax               import *

from modules.geometry.grid      import DOMAIN_SIZE
from modules.thermodynamics.EOS import molecule, speed_of_sound
from jax.numpy                  import stack, ones_like, sin, cos, pi
from jax                        import jit

rho_c, T_c, p_c = molecule.critical_point

@jit
def solution_u_mms_1(mesh, t):
    #set parameters
    k = omega = 2 * pi
    rho0, Arho = 1.2, 0.01
    T0,   AT   = 1.2, 0.01

    c = speed_of_sound(rho0 * rho_c * ones_like(mesh[0]), T0 * T_c * ones_like(mesh[0]))[0]
    vel0, Avel = 0.1, 0.01

    #compute solutions
    rho = rho_c * (rho0 + Arho * cos(k * mesh[0] / DOMAIN_SIZE[0]) * cos(omega * t))
    vel = c *     (vel0 + Avel * sin(k * mesh[0] / DOMAIN_SIZE[0]) * cos(omega * t))
    T   = T_c   * (T0   + AT   * cos(k * mesh[0] / DOMAIN_SIZE[0]) * sin(omega * t))
    return stack((rho, vel, T), axis = 0), 2


@jit
def u_mms_1(mesh):
    #set parameters
    k = omega = 2 * pi
    rho0, Arho = 1.2, 0.01
    T0,   AT   = 1.2, 0.01

    c = speed_of_sound(rho0 * rho_c * ones_like(mesh[0]), T0 * T_c * ones_like(mesh[0]))[0]
    vel0, Avel = 0.1, 0.01

    #compute initial condition
    rho = rho_c * (rho0 + Arho * cos(k * mesh[0] / DOMAIN_SIZE[0]))
    vel = c *     (vel0 + Avel * sin(k * mesh[0] / DOMAIN_SIZE[0]))
    T   = T_c   * (T0 * ones_like(mesh[0]))

    return stack((rho, vel, T), axis = 0), 2 #rvt