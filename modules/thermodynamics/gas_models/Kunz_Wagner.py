"""
The Helmholtz energy and other functions of the Wagner equation

see (referred to as KW-article): The GERG-2004 Wide-Range Equation of State for Natural Gases and Other Mixtures
and: Ideal-Gas Thermodynamic Properties for Natural-Gas Applications
"""

from prep_jax                   import *
from config.conf_thermodynamics import *
from config.conf_geometry       import *

from modules.numerical.computation                              import solve_root_thermo, vectorize_root
from modules.thermodynamics.gas_models.Jaeschke_Schley_ideal    import Jaeschke_Schley
from jax.numpy                                                  import array, zeros_like, exp
from jax.lax                                                    import fori_loop
from jax                                                        import grad, jit, jacfwd


''' check parameter consistency '''

def check_consistency_Wagner():
    pass

''' Derived parameters '''

''' set parameter values for Span-Wagner '''

rho_c, T_c, p_c = molecule.critical_point

#refprop compare
_residual_coeffs = array([
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
])



_temperature_powers = array([
    0.0,    1.25,   1.625,  0.375,
    0.375,  1.375,  1.125,  1.375,
    0.125,  1.625,  3.75,   3.5,
    7.5,    8.0,    6.0,    16.0,
    11.0,   24.0,   26.0,   28.0,
    24.0,   26.0
])

_density_powers = array([
    1.0,    1.0,    2.0,    3.0,
    3.0,    3.0,    4.0,    5.0, 
    6.0,    6.0,    1.0,    4.0,
    1.0,    1.0,    3.0,    3.0,
    4.0,    5.0,    5.0,    5.0,
    5.0,    5.0
])

_exp_powers = array([
    1.0,    1.0,    1.0,
    1.0,    1.0,    1.0,    2.0,
    2.0,    3.0,    3.0,    3.0,
    3.0,    3.0,    5.0,    5.0,
    5.0,    6.0,    6.0
])


''' Helmholtz energy '''
def Kunz_Wagner_ideal(rho, T):
    '''
        computes specific Helmholtz energy of the Jaeschke Schley ideal gas model
    '''
    return Jaeschke_Schley(rho, T)


def Kunz_Wagner_residual(rho, T):
    """
    Computes the residual Helmholtz energy of the Kunz-Wagner real gas equation of state.
    """
    rho_r = rho / rho_c
    T_r   = T_c / T

    ar = zeros_like(rho)

    def body1(i, ar):
        return ar + _residual_coeffs[i] * (rho_r ** _density_powers[i]) * (T_r ** _temperature_powers[i])

    ar = fori_loop(0, 4, body1, ar)

    def body2(i, ar):
        ip = i + 4
        term = (
            _residual_coeffs[ip]
            * (rho_r ** _density_powers[ip])
            * (T_r ** _temperature_powers[ip])
            * exp(-(rho_r ** _exp_powers[i]))
        )
        return ar + term

    ar = fori_loop(0, 18, body2, ar)

    return R_specific * T * ar

def Kunz_Wagner(rho, T):
    """
        Specific Helmholtz energy of the Kunz-Wagner equation of state
    """
    return Kunz_Wagner_ideal(rho, T) + Kunz_Wagner_residual(rho, T)

''' Temperature equation (rho, p) -> T for initial conditions'''
_dAdrho                 = grad(Kunz_Wagner, 0)
_p                      = lambda rho, T: rho**2 * _dAdrho(rho, T)
_root_func_pressure_T   = lambda T, rho, p: p - _p(rho, T)
_drpdT_root             = jacfwd(_root_func_pressure_T, 0)

root_func_pressure_T = vectorize_root(_root_func_pressure_T)
drpdT_root = vectorize_root(_drpdT_root)

def temperature_rpt_Kunz_Wagner(rho, p, Tguess):
    """
        Solve temperature profile from density and pressure for Wagner gas
    """
    return solve_root_thermo(Tguess, rho, p, root_func_pressure_T, drpdT_root, 1e-10, 10)


''' Density equation (p, T) -> rho for initial conditions'''
_root_func_pressure_rho = lambda rho, T, p: p - _p(rho, T)
_drpdrho_root           = jacfwd(_root_func_pressure_rho, 0)

root_func_pressure_rho = vectorize_root(_root_func_pressure_rho)
drpdrho_root = vectorize_root(_drpdrho_root)

def density_ptr_Kunz_Wagner(p, T, rhoguess):
    """
        Solve density profile from pressure and temperature for Wagner gas
    """
    return solve_root_thermo(rhoguess, T, p, root_func_pressure_rho, drpdrho_root, 1e-10, 10)


''' Temperature equations (rho, e) -> T for simulations '''
_dAdT                   = grad(Kunz_Wagner, 1)
_e                      = lambda rho, T: Kunz_Wagner(rho, T) - T * _dAdT(rho, T)
_root_func_energy_T     = lambda T, rho, e: e - _e(rho, T)
_dredT_root             = jacfwd(_root_func_energy_T, 0)

root_func_energy_T = vectorize_root(_root_func_energy_T)
dredT_root = vectorize_root(_dredT_root)

def temperature_ret_Kunz_Wagner(rho, e, Tguess):
    """
        Calculate temperature from density and specific internal energy for Wagner EOS.
    """
    return solve_root_thermo(Tguess, rho, e, root_func_energy_T, dredT_root, 1e-10, 10)

