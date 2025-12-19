'''
Contains functions to convert sets of variables to conservative variables

Conversion functions are added on a as-needed basis
'''

from prep_jax import *
from config.conf_geometry import N_DIMENSIONS

from typing import Tuple

from modules.thermodynamics.EOS     import temperature_rpt, density_ptr, total_energy
from modules.thermodynamics.gas_models.Peng_Robinson import density_ptr_Peng_Robinson as ptr
from modules.thermodynamics.gas_models.Van_der_Waals import temperature_rpt_Van_der_Waals as rpt
from modules.thermodynamics.EOS import Gibbs_energy

def convert(v : jnp.ndarray, vars : int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    match vars:
        case 0: #rvp:
            #density + velocity + pressure (in that order)
            u, T = rvp2u(v)
        case 1: #vpt:
            #velocity + pressure + temperature (in that order)
            u, T = vpt2u(v)
        case 2: #rvt:
            #density + velocity + temperature (in that order)
            u, T = rvt2u(v)
        case _:
            raise ValueError(f"Unknown variable set: {vars}")
    return u, T

@jax.jit
def rvp2u(v : jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    rho = v[0]
    vel = v[1:(N_DIMENSIONS+1), ...]
    p   = v[N_DIMENSIONS + 1]

    Tguess  = rpt(rho, p, None)
    T       = temperature_rpt(rho, p, Tguess)

    m = vel * rho
    E = total_energy(rho, T, vel) 
    
    if N_DIMENSIONS == 1:
        u = jnp.stack((rho, m[0], E), axis = 0)  
    elif N_DIMENSIONS == 2: 
        u  = jnp.stack((rho, m[0], m[1], E), axis=0) 
    elif N_DIMENSIONS == 3:
        u  = jnp.stack((rho, m[0], m[1], m[2], E), axis=0) 

    return u, T

@jax.jit
def vpt2u(v : jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    vel = v[:N_DIMENSIONS, ...]
    p   = v[N_DIMENSIONS, ...]
    T   = v[N_DIMENSIONS + 1, ...]

    rhoguess    = ptr(p, T, None)
    rho         = density_ptr(p, T, rhoguess)
    m           = vel * rho
    E           = total_energy(rho, T, vel)

    if N_DIMENSIONS == 1:
        u = jnp.stack((rho, m[0], E), axis = 0) 
    elif N_DIMENSIONS == 2: 
        u  = jnp.stack((rho, m[0], m[1], E), axis=0) 
    elif N_DIMENSIONS == 3:
        u  = jnp.stack((rho, m[0], m[1], m[2], E), axis=0) 

    return u, T

@jax.jit
def rvt2u(v : jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    rho = v[0, ...]
    vel = v[1:N_DIMENSIONS + 1, ...]
    T   = v[N_DIMENSIONS + 1, ...]

    m = vel * rho
    E = total_energy(rho, T, vel)

    if N_DIMENSIONS == 1:
        u = jnp.stack((rho, m[0], E), axis = 0) 
    elif N_DIMENSIONS == 2: 
        u  = jnp.stack((rho, m[0], m[1], E), axis=0) 
    elif N_DIMENSIONS == 3:
        u  = jnp.stack((rho, m[0], m[1], m[2], E), axis=0) 

    return u, T

@jax.jit
def entropy_variables(u : jnp.ndarray, T : jnp.ndarray) -> jnp.ndarray:
    g = Gibbs_energy(u[0], T)
    v = u[1:(N_DIMENSIONS+1)] / u[0]

    eta_rho = (g - 1/2 * jnp.sum(v**2, axis = 0)) / T
    eta_m = v / T
    eta_E = - 1.0 / T

    if N_DIMENSIONS == 1:
        eta = jnp.stack((eta_rho, eta_m[0], eta_E), axis = 0) 
    elif N_DIMENSIONS == 2: 
        eta  = jnp.stack((eta_rho, eta_m[0], eta_m[1], eta_E), axis=0) 
    elif N_DIMENSIONS == 3:
        eta  = jnp.stack((eta_rho, eta_m[0], eta_m[1], eta_m[2], eta_E), axis=0) 

    return eta
