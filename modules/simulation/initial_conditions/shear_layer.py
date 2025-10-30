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
from modules.thermodynamics.EOS import density_ptr, T_c

def chan_shear_layer_2d(mesh, rho_c, p_c):
    """
    Generate Chan's shear layer experiment
    """

    #shear layer thickness parameters
    sig = 0.15
    alpha = 0.1
    sig2 = sig**2

    #velocity difference between layers
    du0 = 100 #Mach order 0.1

    #number of roll ups
    k = 1

    x_trans = 2 * mesh[0] - 1
    y_trans = 2 * mesh[1] - 1
    yp = y_trans + 1/2
    ym = y_trans - 1/2

    rho = rho_c * (1 + 1 / (1 + jnp.exp(-yp/sig2)) - 1 / (1 + jnp.exp(-ym/sig2)))
    u = du0 * (1 / (1 + jnp.exp(-yp/sig2)) - 1 / (1 + jnp.exp(-ym/sig2)) - 1/2)
    v = alpha * du0 * jnp.sin(k * jnp.pi * x_trans) * (1/(1+jnp.exp(-yp/sig2)) - 1/(1+jnp.exp(-ym/sig2)))
    p = 2.0 * p_c * jnp.ones_like(mesh[0])
    
    return jnp.stack((rho, u, v, p), axis = 0)

def coppola_shear_layer_2d(mesh, rho_c, p_c):
    """
    Generate Coppola's shear layer (edited)

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

    u0 = 100 #m/s

    u = u0 * jnp.where(mesh_y_trans > 0.0, (1 - A * jnp.tanh(mesh_y_trans / delta)), (1 + A * jnp.tanh(mesh_y_trans / delta)))
    v = eps * u0 * jnp.sin(2.0 * k * jnp.pi / DOMAIN_SIZE[0] * mesh_x_trans) * jnp.exp(-4.0 * mesh_y_trans**2 / delta)
    T = T0 * jnp.where(mesh_y_trans > 0, (1 + B * jnp.tanh(mesh_y_trans / delta)), (1 - B * jnp.tanh(mesh_y_trans/ delta)))
    p = 2 * p_c * jnp.ones_like(mesh_x_trans)
    rho = density_ptr(p, T, 2 * rho_c * jnp.ones_like(T))
    
    return jnp.stack((rho, u, v, p), axis = 0)

def bernades_shear_layer_2d(mesh, rho_c, p_c):
    """
    Generate Bernades's shear layer

    Lx = 1.0, Ly = 0.5
    """

    u0 = 25
    At = 3/8
    Au = 3/8
    delta_bar = 20.0

    k = 2
    ag = 0.1
    A = 0.1
    delta_u = 10.0
    ubw = 20.0

    u_base = jnp.where( mesh[1] < 0.5,
                        u0 * (1 + Au * jnp.tanh(delta_bar * (mesh[1] - DOMAIN_SIZE[1]/4))),
                        u0 * (1 - Au * jnp.tanh(delta_bar * (mesh[1] - 3*DOMAIN_SIZE[1]/4)))
    )
    v_base = jnp.zeros_like(u_base)
    T = jnp.where( mesh[1] < 0.5,
                        T_c * (3 * At - At * jnp.tanh(delta_bar * (mesh[1] - DOMAIN_SIZE[1]/4))),
                        T_c * (3 * At + At * jnp.tanh(delta_bar * (mesh[1] - 3*DOMAIN_SIZE[1]/4)))
    )
    p = 2 * p_c * jnp.ones_like(mesh[0])

    #integral-calculator.com with (1 + 3/8 * tanh(y) - 4/5)/(2/5)*(1 - ((1 + 3/8 *tanh(y) - 4/5)/(2/5)) from -1/4 to 1/4
    theta = (19 * jnp.sqrt(jnp.exp(1)) - 31) / (2 * jnp.sqrt(jnp.exp(1)) + 2)

    ay = jnp.where( mesh[1] < 0.5,
                    delta_u * (jnp.exp(-(mesh[1] - DOMAIN_SIZE[1]/2)**2 / theta) + jnp.exp(-(mesh[1] - DOMAIN_SIZE[1]/4)**2 / theta) + jnp.exp(-(mesh[1])**2 / theta)),
                    delta_u * (jnp.exp(-(mesh[1] - DOMAIN_SIZE[1])**2 / theta) + jnp.exp(-(mesh[1] - 3*DOMAIN_SIZE[1]/4)**2 / theta) + jnp.exp(-(mesh[1] - DOMAIN_SIZE[1]/2)**2 / theta)),
    )

    key = jax.random.PRNGKey(0)
    key_g, _ = jax.random.split(key)
    gx = jax.random.uniform(key_g, shape=(GRID_RESOLUTION[0],), minval=-0.5, maxval=0.5)

    base_mode = jnp.sin(k * jnp.pi * mesh[0]) + ag * gx[:,None]
    Up = ay * base_mode
    Vp = ay * base_mode

    u = u_base + A * Up
    v = v_base + A * Vp

    rho = density_ptr(p, T, 2 * rho_c * jnp.ones_like(T))

    return jnp.stack((rho, u, v, p), axis = 0)
