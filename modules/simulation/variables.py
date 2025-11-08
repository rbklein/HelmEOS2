'''
Contains functions to convert sets of variables to conservative variables

Conversion functions are added on a as-needed basis
'''

from prep_jax import *
from config.conf_geometry import N_DIMENSIONS

from typing import Tuple

from modules.thermodynamics.EOS     import temperature_rpt, total_energy
from modules.thermodynamics.gas_models.Van_der_Waals import temperature_rpt_Van_der_Waals as rpt

def convert(v : jnp.ndarray, vars : str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    match vars:
        case 'rvp':
            #density + velocity + pressure (in that order)
            u, T = rvp2u(v)

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
    if N_DIMENSIONS == 2: 
        u  = jnp.stack((rho, m[0], m[1], E), axis=0) 
    elif N_DIMENSIONS == 3:
        u  = jnp.stack((rho, m[0], m[1], m[2], E), axis=0) 

    return u, T


