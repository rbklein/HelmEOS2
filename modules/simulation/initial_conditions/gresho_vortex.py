"""
Docstring for modules.simulation.initial_conditions.gresho_vortex
"""

from prep_jax import *
from config.conf_geometry       import *
from config.conf_thermodynamics import *

from jax.numpy  import where, logical_and, sqrt, ones_like, log, stack
from jax        import jit

rho_c, T_c, p_c = molecule.critical_point

def u_theta(r):
    u1 = where(logical_and(0 <= r, r < 0.2), 5 * r, 0)
    u2 = where(logical_and(0.2 <= r, r < 0.4), 2 - 5 * r, 0)
    return u1 + u2

def p_r(r):
    p0 = 1.3 * p_c * ones_like(r)
    p1 = where(logical_and(0 <= r, r < 0.2), 25/2 * r**2, 0)
    p2 = where(logical_and(0.2 <= r, r < 0.4), 4 * log(r / 0.2) - 20 * r + 4 + 25/2 * r**2, 0)
    p3 = where(r >= 0.4, 4 * log(2) - 2, 0)
    return p1 + p2 + p3 + p0

@jit
def gresho_vortex(mesh):
    """
    Domain [0, 1] x [0, 1]
    """
    U_inf = 1.0 

    rho = 1.1 * rho_c * ones_like(mesh[0])

    r = sqrt((mesh[0] - 0.5)**2 + (mesh[1] - 0.5)**2)
    u = U_inf - u_theta(r) * (mesh[1] - 0.5) / r
    v = u_theta(r) * (mesh[0] - 0.5) / r

    p = p_r(r)

    return stack((rho, u, v, p), axis = 0), 0