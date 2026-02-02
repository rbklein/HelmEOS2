"""
    Implementation of a constant dynamic viscosity
"""

from prep_jax                   import *
from config.conf_thermodynamics import *

from jax.numpy import ones_like

''' Functions '''

def check_consistency():
    """
        consistency checks for constant dynamic viscosity
    """
    assert "value" in VISC_DYN_parameters, "Constant dynamic viscosity must be assigned a value"  
    assert isinstance(VISC_DYN_parameters["value"], (float, int)), "Constant dynamic viscosity value must be a floating point value or an integer"
    assert VISC_DYN_parameters["value"] >= 0.0, "Constant dynamic viscosity must be non-negative"

def constant_dynamic_viscosity(u, T):
    """
        returns a user-defined constant dynamic viscosity for every value in the temperature field
    """
    return VISC_DYN_parameters["value"] * ones_like(T)
