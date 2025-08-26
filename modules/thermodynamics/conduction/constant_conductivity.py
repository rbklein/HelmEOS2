"""
    Implements a constant thermal conductivity
"""

from prep_jax import *
from config.conf_thermodynamics import *

''' Functions '''

def constant_thermal_conductivity(T):
    """
    Implements a constant thermal conductivity
    """
    return THERMAL_COND_parameters["value"] * jnp.ones_like(T)

