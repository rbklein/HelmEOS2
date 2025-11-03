"""
The Helmholtz energy and other functions of the Span-Wagner equation
"""

from prep_jax import *
from config.conf_thermodynamics import *
from config.conf_geometry import *

from modules.numerical.computation import solve_root_thermo, vectorize_root

''' check parameter consistency '''

def check_consistency_Span_Wagner():
    pass

''' Derived parameters '''

R_specific = UNIVERSAL_GAS_CONSTANT / MOLAR_MASS #J K^-1 kg^-1 specific gas constant

''' set parameter values for Span-Wagner '''

rho_c, T_c, p_c = molecule.critical_points


''' Helmholtz energy '''

def Span_Wagner(rho, T):
    """
        Specific Helmholtz energy of the Span-Wagner equation of state
    """
    gamma = 1.2
    return - R_specific * T * (1 + jnp.log(T**(1/(gamma - 1)) / rho))


''' Temperature equation (rho, p) -> T for initial conditions'''
_dAdrho                 = jax.grad(Span_Wagner, 0)
_p                      = lambda rho, T: rho**2 * _dAdrho(rho, T)
_root_func_pressure_T   = lambda T, rho, p: p - _p(rho, T)
_drpdT_root             = jax.grad(_root_func_pressure_T, 0)

root_func_pressure_T = vectorize_root(_root_func_pressure_T)
drpdT_root = vectorize_root(_drpdT_root)

def temperature_rpt_Span_Wagner(rho, p, Tguess):
    """
        Solve temperature profile from density and pressure for Span-Wagner gas
    """
    return solve_root_thermo(Tguess, rho, p, root_func_pressure_T, drpdT_root, 1e-10, 10)


''' Density equation (p, T) -> rho for initial conditions'''
_root_func_pressure_rho = lambda rho, T, p: p - _p(rho, T)
_drpdrho_root           = jax.grad(_root_func_pressure_rho, 0)

root_func_pressure_rho = vectorize_root(_root_func_pressure_rho)
drpdrho_root = vectorize_root(_drpdrho_root)

def density_ptr_Span_Wagner(p, T, rhoguess):
    """
        Solve density profile from pressure and temperature for Span-Wagner gas
    """
    return solve_root_thermo(rhoguess, T, p, root_func_pressure_rho, drpdrho_root, 1e-10, 10)


''' Temperature equations (rho, e) -> T for simulations '''
_dAdT                   = jax.grad(Span_Wagner, 1)
_e                      = lambda rho, T: Span_Wagner(rho, T) - T * _dAdT(rho, T)
_root_func_energy_T     = lambda T, rho, e: e - _e(rho, T)
_dredT_root             = jax.grad(_root_func_energy_T, 0)

root_func_energy_T = vectorize_root(_root_func_energy_T)
dredT_root = vectorize_root(_dredT_root)

def temperature_ret_Span_Wagner(rho, e, Tguess):
    """
        Calculate temperature from density and specific internal energy for Span-Wagner EOS.
    """
    return solve_root_thermo(Tguess, rho, e, root_func_energy_T, dredT_root, 1e-10, 10)

