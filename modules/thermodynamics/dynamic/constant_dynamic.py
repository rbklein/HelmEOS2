"""
    Implementation of a constant dynamic viscosity
"""

from prep_jax import *
from config.conf_thermodynamics import *

''' Functions '''

def constant_dynamic_viscosity(T):
    """
        returns a user-defined constant dynamic viscosity for every value in the temperature field
    """
    return VISC_DYN_parameters["value"] * jnp.ones_like(T)
