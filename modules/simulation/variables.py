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

def convert(v : jnp.ndarray, vars : str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    match vars:
        case 'rvp':
            #density + velocity + pressure (in that order)
            u, T = rvp2u(v)
        case 'vpt':
            #velocity + pressure + temperature (in that order)
            u, T = vpt2u(v)
        case 'rvt':
            #density + velocity + temperature (in that order)
            u, T = rvt2u(v)
        case _:
            raise ValueError(f"Unknown variable set: {vars}")
    return u, T


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

