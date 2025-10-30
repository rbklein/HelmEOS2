"""
The Helmholtz energy and other functions of the ideal gas law
"""

from prep_jax import *
from config.conf_thermodynamics import *

''' check parameter consistency '''

def check_consistency_ideal():
    assert "gamma" in EOS_parameters, "EOS parameters for ideal gas must include 'gamma'"
    assert isinstance(EOS_parameters["gamma"], (float, int)), "EOS parameter 'gamma' must be a number"
    assert EOS_parameters["gamma"] > 1, "EOS parameter 'gamma' must be greater than 1"

''' Derived parameters '''

R_specific = UNIVERSAL_GAS_CONSTANT / MOLAR_MASS #J K^-1 kg^-1 specific gas constant

''' set critical point values for ideal gas (does not have a critical point)'''

#Critical values for Nitrogen, to use for initial conditions
rho_c, T_c, p_c = molecule.critical_points

''' Helmholtz Energies '''

def ideal_gas(rho, T):
    """
        Specific Helmholtz energy of the ideal gas equation of state

        f = 2 / (gamma - 1)
    """
    gamma = EOS_parameters["gamma"]

    return - R_specific * T * (1 + jnp.log(T**(1/(gamma - 1)) / rho))


''' Temperature equation (rho, p) -> T for initial conditions'''

def temperature_rpt_ideal(rho, p, Tguess):
    """
        Solve temperature profile from density and pressure for ideal gas
    """
    T = p / (rho * R_specific)
    return T


''' Density equation (p, T) -> rho for initial conditions'''

def density_ptr_ideal(p, T, rho_guess):
    """
        Solve density profile from temperature and pressure for ideal gas
    """
    rho = p / (R_specific * T)
    return rho


''' Temperature equations (rho, e) -> T for simulations '''

def temperature_ret_ideal(rho, e, Tguess):
    """
        Calculate temperature from density and specific internal energy for ideal gas EOS.
    """
    gamma = EOS_parameters["gamma"]
    T = ((gamma - 1) / R_specific) * e
    return T

