"""
Initial conditions for several shear layer initial conditions

Chan's shear layer:
article: Entropy stable reduced order modelling of nonlinear conservation laws
authors: J. Chan
"""

from prep_jax import *
from modules.geometry.grid import *
from modules.thermodynamics.EOS import *

rho_c, T_c, p_c = molecule.critical_point

@jax.jit
def chan_shear_layer_2d(mesh):
    """
    Generate Chan's shear layer experiment

    Domain  : [0, 1]^2
    T       : 0.02
    """

    #shear layer thickness parameters
    sig = 0.15
    alpha = 0.1
    sig2 = sig**2

    #velocity difference between layers
    du0 = 10 #Mach order 0.1

    #number of roll ups
    k = 1

    x_trans = 2 * mesh[0] - 1
    y_trans = 2 * mesh[1] - 1
    yp = y_trans + 1/2
    ym = y_trans - 1/2

    amp = 0.4
    base = 0.8

    rho = rho_c * (base + amp * (1 / (1 + jnp.exp(-yp/sig2)) - 1 / (1 + jnp.exp(-ym/sig2))))
    u = du0 * (amp * (1 / (1 + jnp.exp(-yp/sig2)) - 1 / (1 + jnp.exp(-ym/sig2))) - 1/2)
    v = alpha * du0 * jnp.sin(k * jnp.pi * x_trans) * amp * (1/(1+jnp.exp(-yp/sig2)) - 1/(1+jnp.exp(-ym/sig2)))
    p = 1.2 * p_c * jnp.ones_like(mesh[0])
    
    return jnp.stack((rho, u, v, p), axis = 0), 0 #rvp

