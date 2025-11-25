"""
Taylor-Green vortex initial conditions
"""

from prep_jax import *
from modules.geometry.grid import *
from modules.thermodynamics.EOS import *

rho_c, T_c, p_c = molecule.critical_points

@jax.jit
def Taylor_Green_vortex_3d(mesh):
    """
        Domain : [- pi L, pi L]^3
        T      : ???
    """
    #plural?

    rho0    = 1.5 * rho_c
    T0      = 1.1 * T_c

    # Determine p0, c0 from rho0 and T0
    p0      = pressure(rho0 * jnp.ones_like(mesh[0]), T0 * jnp.ones_like(mesh[0]))[0,0,0]
    c0      = speed_of_sound(rho0 * jnp.ones_like(mesh[0]), T0 * jnp.ones_like(mesh[0]))[0,0,0]
    U0      = 0.1 * c0

    #print('ref vel: ', U0)

    L       = DOMAIN_SIZE[0] / (2 * jnp.pi)
    X, Y, Z = mesh[0] - jnp.pi * L, mesh[1] - jnp.pi * L, mesh[2] - jnp.pi * L

    p = p0 + rho0 * U0**2 / 16.0 * (jnp.cos(2 * X / L) + jnp.cos(2 * Y / L)) * (jnp.cos(2 * Z / L) + 2)
    T = T0 * jnp.ones_like(p)
    u = U0 * jnp.sin(X / L) * jnp. cos(Y / L) * jnp.sin(Z / L)
    v = - U0 * jnp.cos(X / L) * jnp.sin(Y / L) * jnp.sin(Z / L)
    w = jnp.zeros_like(u)

    return jnp.stack((u, v, w, p, T), axis = 0), 1 #vpt
