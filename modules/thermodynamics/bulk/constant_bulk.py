"""
    Implementation of a constant bulk viscosity
"""

from prep_jax                   import *
from config.conf_thermodynamics import *

from jax.numpy import ones_like

''' Functions '''

def check_consistency():
    """
        consistency checks for constant bulk viscosity
    """
    assert "value" in VISC_BULK_parameters, "Constant bulk viscosity must be assigned a value"
    assert isinstance(VISC_BULK_parameters["value"], (float, int)), "Constant bulk viscosity value must be a floating point value or an integer"
    assert VISC_BULK_parameters["value"] >= 0.0, "Constant bulk viscosity must be non-negative"

def constant_bulk_viscosity(u, T):
    """
        returns a user-defined constant bulk viscosity for every value in the temperature field
    """
    return VISC_BULK_parameters["value"] * ones_like(T)