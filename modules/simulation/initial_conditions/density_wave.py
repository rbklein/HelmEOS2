"""
Density Wave initial conditions
"""

from prep_jax               import *

from modules.geometry.grid      import DOMAIN_SIZE
from modules.thermodynamics.EOS import molecule, pressure
from modules.numerical.computation import evaluate_scalar_thermo
from jax.numpy                  import stack, ones_like, sin, pi
from jax                        import jit

rho_c, T_c, p_c = molecule.critical_point

@jit
def density_wave_1d(mesh):
    """
    Generate a 1D density wave pattern.
    
    Returns:
    jnp.ndarray: A stack of 2D arrays representing the density wave initial condition.
    """

    T0   = 1.5 * T_c
    p0   = 1.01 * p_c #evaluate_scalar_thermo(pressure, rho0, T0)

    u = 10 * ones_like(mesh[0]) 
    p = p0 * ones_like(mesh[0])  
    T = T0 + 0.499999 * T_c * sin(2 * pi * mesh[0] / DOMAIN_SIZE[0])


    return stack((u, p, T), axis=0), 1 #rvp  # Stack to create a 2D array with shape (4, n_x, n_y)