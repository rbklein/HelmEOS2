"""
    Implements a constant thermal conductivity
"""

from prep_jax import *
from config.conf_thermodynamics import *

''' Functions '''

def check_consistency():
    """
        consistency checks for constant thermal conductivity
    """
    assert "value" in THERMAL_COND_parameters, "Constant thermal conductivity must be assigned a value"
    assert isinstance(THERMAL_COND_parameters["value"], (float, int)), "Constant thermal conductivity value must be a floating point value or an integer"
    assert THERMAL_COND_parameters["value"] >= 0.0, "Constant bulk viscosity must be non-negative"

def constant_thermal_conductivity(T):
    """
    Implements a constant thermal conductivity
    """
    return THERMAL_COND_parameters["value"] * jnp.ones_like(T)

