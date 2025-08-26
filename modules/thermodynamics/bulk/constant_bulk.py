"""
    Implementation of a constant bulk viscosity
"""

from prep_jax import *
from config.conf_thermodynamics import *

''' Functions '''

def constant_bulk_viscosity(T):
    """
        returns a user-defined constant bulk viscosity for every value in the temperature field
    """
    return VISC_BULK_parameters["value"] * jnp.ones_like(T)