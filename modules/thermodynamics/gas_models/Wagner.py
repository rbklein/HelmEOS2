"""
The Helmholtz energy and other functions of the Wagner equation

see (referred to as KW-article): The GERG-2004 Wide-Range Equation of State for Natural Gases and Other Mixtures
and: Ideal-Gas Thermodynamic Properties for Natural-Gas Applications
"""

from prep_jax import *
from config.conf_thermodynamics import *
from config.conf_geometry import *

from modules.numerical.computation import solve_root_thermo, vectorize_root
from modules.thermodynamics.gas_models.Jaeschke_Schley_ideal import Jaeschke_Schley
#from modules.thermodynamics.gas_models.ideal_gas import ideal_gas

''' check parameter consistency '''

def check_consistency_Wagner():
    pass

''' Derived parameters '''

R_specific = UNIVERSAL_GAS_CONSTANT / MOLAR_MASS #J K^-1 kg^-1 specific gas constant

''' set parameter values for Span-Wagner '''

rho_c, T_c, p_c = molecule.critical_point

#refprop compare
_residual_coeffs = [
    0.52646564804653,
    -1.4995725042592,
    0.27329786733782,
    0.12949500022786,
    0.15404088341841,
    -0.58186950946814,
    -0.18022494838296,
    -0.095389904072812,
    -0.0080486819317679,
    -0.03554775127309,
    -0.28079014882405,
    -0.082435890081677,
    0.010832427979006,
    -0.0067073993161097,
    -0.0046827907600524,
    -0.028359911832177,
    0.019500174744098,
    -0.21609137507166,
    0.43772794926972,
    -0.22130790113593,
    0.015190189957331,
    -0.0153809489533
]



_temperature_powers = [
    0.0,    1.25,   1.625,  0.375,
    0.375,  1.375,  1.125,  1.375,
    0.125,  1.625,  3.75,   3.5,
    7.5,    8.0,    6.0,    16.0,
    11.0,   24.0,   26.0,   28.0,
    24.0,   26.0
]

_density_powers = [
    1.0,    1.0,    2.0,    3.0,
    3.0,    3.0,    4.0,    5.0, 
    6.0,    6.0,    1.0,    4.0,
    1.0,    1.0,    3.0,    3.0,
    4.0,    5.0,    5.0,    5.0,
    5.0,    5.0
]

_exp_powers = [
    1.0,    1.0,    1.0,
    1.0,    1.0,    1.0,    2.0,
    2.0,    3.0,    3.0,    3.0,
    3.0,    3.0,    5.0,    5.0,
    5.0,    6.0,    6.0
]

# _residual_coeffs     = jnp.array(_residual_coeffs)
# _temperature_powers  = jnp.array(_temperature_powers)
# _density_powers      = jnp.array(_density_powers)
# _exp_powers          = jnp.array(_exp_powers)


''' Helmholtz energy '''
def Wagner_ideal(rho, T):
    '''
        computes specific Helmholtz energy of the Jaeschke Schley ideal gas model
    '''
    return Jaeschke_Schley(rho, T)

def Wagner_residual(rho, T):
    '''
        computes the residual Helmholtz energy of the Kunz-Wagner real gas equation of state
    '''
    ar     = jnp.zeros_like(rho)
    rho_r   = rho / rho_c
    T_r     = T_c / T

    for i in range(4):
        ar += _residual_coeffs[i] * (rho_r**_density_powers[i]) * (T_r**_temperature_powers[i])

    for i in range(18):
        ip = i + 4
        ar += _residual_coeffs[ip] * (rho_r**_density_powers[ip]) * (T_r**_temperature_powers[ip]) * jnp.exp(-rho_r**_exp_powers[i])

    return R_specific * T * ar

def Wagner(rho, T):
    """
        Specific Helmholtz energy of the Kunz-Wagner equation of state
    """
    return Wagner_ideal(rho, T) + Wagner_residual(rho, T)


''' Temperature equation (rho, p) -> T for initial conditions'''
_dAdrho                 = jax.grad(Wagner, 0)
_p                      = lambda rho, T: rho**2 * _dAdrho(rho, T)
_root_func_pressure_T   = lambda T, rho, p: p - _p(rho, T)
_drpdT_root             = jax.grad(_root_func_pressure_T, 0)

root_func_pressure_T = vectorize_root(_root_func_pressure_T)
drpdT_root = vectorize_root(_drpdT_root)

def temperature_rpt_Wagner(rho, p, Tguess):
    """
        Solve temperature profile from density and pressure for Wagner gas
    """
    return solve_root_thermo(Tguess, rho, p, root_func_pressure_T, drpdT_root, 1e-10, 100)


''' Density equation (p, T) -> rho for initial conditions'''
_root_func_pressure_rho = lambda rho, T, p: p - _p(rho, T)
_drpdrho_root           = jax.grad(_root_func_pressure_rho, 0)

root_func_pressure_rho = vectorize_root(_root_func_pressure_rho)
drpdrho_root = vectorize_root(_drpdrho_root)

def density_ptr_Wagner(p, T, rhoguess):
    """
        Solve density profile from pressure and temperature for Wagner gas
    """
    return solve_root_thermo(rhoguess, T, p, root_func_pressure_rho, drpdrho_root, 1e-10, 10)


''' Temperature equations (rho, e) -> T for simulations '''
_dAdT                   = jax.grad(Wagner, 1)
_e                      = lambda rho, T: Wagner(rho, T) - T * _dAdT(rho, T)
_root_func_energy_T     = lambda T, rho, e: e - _e(rho, T)
_dredT_root             = jax.grad(_root_func_energy_T, 0)

root_func_energy_T = vectorize_root(_root_func_energy_T)
dredT_root = vectorize_root(_dredT_root)

def temperature_ret_Wagner(rho, e, Tguess):
    """
        Calculate temperature from density and specific internal energy for Wagner EOS.
    """
    return solve_root_thermo(Tguess, rho, e, root_func_energy_T, dredT_root, 1e-10, 10)

