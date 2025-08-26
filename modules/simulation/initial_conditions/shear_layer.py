"""
Initial conditions for several shear layer initial conditions

Chan's shear layer:
article: Entropy stable reduced order modelling of nonlinear conservation laws
authors: J. Chan

Coppola's shear layer:
article: Entropy conservative discretization of compressible Euler equations with an arbitrary equation of state
authors: A. Aiello, C. De Michele, G. 


"""

from prep_jax import *
from modules.geometry.grid import *
from modules.thermodynamics.EOS import density_eos

def chan_shear_layer_2d(mesh, rho_c, p_c):
    """
    Generate Chan's shear layer experiment
    """

    #shear layer thickness parameters
    sig = 0.15
    alpha = 1.0
    sig2 = sig**2

    #velocity difference between layers
    du0 = 10

    #number of roll ups
    k = 1

    x_trans = 2 * mesh[0] - 1
    y_trans = 2 * mesh[1] - 1
    yp = y_trans + 1/2
    ym = y_trans - 1/2

    rho = rho_c * (1 + 1 / (1 + jnp.exp(-yp/sig2)) - 1 / (1 + jnp.exp(-ym/sig2)))
    u = du0 * (1 / (1 + jnp.exp(-yp/sig2)) - 1 / (1 + jnp.exp(-ym/sig2)) - 1/2)
    v = alpha * jnp.sin(k * jnp.pi * x_trans) * (1/(1+jnp.exp(-yp/sig2)) - 1/(1+jnp.exp(-ym/sig2)))
    p = 2.0 * p_c * jnp.ones_like(mesh[0])
    
    return jnp.stack((rho, u, v, p), axis = 0)

def coppola_shear_layer_2d(mesh, rho_c, p_c):
    """
    Generate Coppola's shear layer

    domain: Lx = .5 m, Ly = .25 m, [-Lx, Lx] x [-Ly, Ly]
    """

    mesh_x_trans = mesh[0] - 0.5
    mesh_y_trans = mesh[1] - 0.25

    A = 3.0 / 8.0 
    B = 3.0 / 8.0
    delta = 1.0 / 15.0
    eps = 0.1
    k = 3.0
    T0 = 110

    u0 = 20 #m/s
    u = u0 * jnp.where(mesh_y_trans > 0.0, (1 - A * jnp.tanh(mesh_y_trans / delta)), (1 + A * jnp.tanh(mesh_y_trans / delta)))
    v = eps * jnp.sin(2.0 * k * jnp.pi / DOMAIN_SIZE[0] * mesh_x_trans) * jnp.exp(-4.0 * mesh_y_trans**2 / delta)
    T = T0 * jnp.where(mesh_y_trans > 0, (1 + B * jnp.tanh(mesh_y_trans / delta)), (1 - B * jnp.tanh(mesh_y_trans/ delta)))
    p = 2 * p_c * jnp.ones_like(mesh_x_trans)
    rho = density_eos(p, T)
    
    return jnp.stack((rho, u, v, p), axis = 0)