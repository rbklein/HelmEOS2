"""
The Helmholtz energy and other functions of the Wagner equation

see (referred to as KW-article): The GERG-2004 Wide-Range Equation of State for Natural Gases and Other Mixtures
and: Ideal-Gas Thermodynamic Properties for Natural-Gas Applications
"""

from prep_jax                   import *
from config.conf_thermodynamics import *
from config.conf_geometry       import *

from modules.numerical.computation                              import solve_root_thermo
from modules.thermodynamics.gas_models.Jaeschke_Schley_ideal    import Jaeschke_Schley, Jaeschke_Schley_drho, Jaeschke_Schley_dT, Jaeschke_Schley_drho2, Jaeschke_Schley_dT2, Jaeschke_Schley_drhodT
from jax.numpy                                                  import array, zeros_like, exp
from jax.lax                                                    import fori_loop


''' check parameter consistency '''

def check_consistency_Wagner():
    pass

''' Derived parameters '''

''' set parameter values for Span-Wagner '''

rho_c, T_c, p_c = molecule.critical_point

#refprop compare
_cr = array([
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



_ct = array([
    0.0,    1.25,   1.625,  0.375,
    0.375,  1.375,  1.125,  1.375,
    0.125,  1.625,  3.75,   3.5,
    7.5,    8.0,    6.0,    16.0,
    11.0,   24.0,   26.0,   28.0,
    24.0,   26.0
])

_crho = array([
    1.0,    1.0,    2.0,    3.0,
    3.0,    3.0,    4.0,    5.0, 
    6.0,    6.0,    1.0,    4.0,
    1.0,    1.0,    3.0,    3.0,
    4.0,    5.0,    5.0,    5.0,
    5.0,    5.0
])

_ce = array([
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
        return ar + _cr[i] * (rho_r ** _crho[i]) * (T_r ** _ct[i])

    ar = fori_loop(0, 4, body1, ar)

    def body2(i, ar):
        ip = i + 4
        term = (
            _cr[ip]
            * (rho_r ** _crho[ip])
            * (T_r ** _ct[ip])
            * exp(-(rho_r ** _ce[i]))
        )
        return ar + term

    ar = fori_loop(0, 18, body2, ar)

    return R_specific * T * ar


def Kunz_Wagner(rho, T):
    """
        Specific Helmholtz energy of the Kunz-Wagner equation of state
    """
    return Kunz_Wagner_ideal(rho, T) + Kunz_Wagner_residual(rho, T)




''' Manual residual derivatives '''

def dresidual_drho(rho, T):
    """
    Manually computed density derivative of residual Helmholtz energy for performance critical situations
    """
    rho_r = rho / rho_c
    T_r   = T_c / T

    ar = zeros_like(rho)

    def body1(i, ar):
        c = _cr[i] * (T_r ** _ct[i]) * (1 / rho_c)
        return ar + c * (_crho[i] * rho_r ** (_crho[i] - 1.0))

    ar = fori_loop(0, 4, body1, ar)

    def body2(i, ar):
        ip = i + 4
        c = _cr[ip] * (T_r ** _ct[ip]) * (1 / rho_c)
        return ar + c * (- rho_r**(_crho[ip] - 1)) * (_ce[i] * rho_r**_ce[i] - _crho[ip]) * exp(- rho_r ** _ce[i])

    ar = fori_loop(0, 18, body2, ar)

    return R_specific * T * ar


def d2residual_drho2(rho, T):
    """
    Manually computed second density derivative of residual Helmholtz energy for performance critical situations
    """
    rho_r = rho / rho_c
    T_r = T_c / T

    ar = zeros_like(rho)

    def body1(i, ar):
        c = _cr[i] * (T_r ** _ct[i]) * (1 / rho_c)**2
        return ar + c * (_crho[i] * (_crho[i] - 1.0) * (rho_r ** (_crho[i] - 2.0)))
    
    ar = fori_loop(0, 4, body1, ar)

    def body2(i, ar):
        """
        d^2/d^2x (f(y(x))) = f''(y(x)) * (y'(x))^2 + f'(y(x)) * (y''(x))
        """
        ip = i + 4
        c = _cr[ip] * (T_r ** _ct[ip]) * (1 / rho_c)**2
        term1 = (_ce[i]**2) * rho_r**(2 * _ce[i])
        term2 = ((1 - 2 * _crho[ip]) * _ce[i] - _ce[i]**2) * rho_r**_ce[i]
        term3 = _crho[ip]**2 - _crho[ip]
        return ar + c * (rho_r ** (_crho[ip] - 2)) * (term1 + term2 + term3) * (exp(-rho_r**_ce[i]))
    
    ar = fori_loop(0, 18, body2, ar)

    return R_specific * T * ar



def dresidual_dT(rho, T):
    """
    Manually computed temperature derivative of residual Helmholtz energy for performance critical situations
    """
    rho_r = rho / rho_c
    T_r = T_c / T

    ar = zeros_like(rho)
    def body1(i, ar):
        c = _cr[i] * (rho_r ** _crho[i])
        return ar + c * (1 - _ct[i]) * (T_r ** _ct[i])
    
    ar = fori_loop(0, 4, body1, ar)

    def body2(i, ar):
        ip = i + 4
        c = (
            _cr[ip]
            * (rho_r ** _crho[ip])
            * exp(-(rho_r ** _ce[i]))
        )
        return ar + c * (1 - _ct[ip]) * (T_r ** _ct[ip])

    ar = fori_loop(0, 18, body2, ar)

    return R_specific * ar


def d2residual_dT2(rho, T):
    """
    Manually computed second temperature derivative of residual Helmholtz energy for performance critical situations
    """
    rho_r = rho / rho_c
    T_r = T_c / T

    ar = zeros_like(rho)
    def body1(i, ar):
        c = _cr[i] * (rho_r ** _crho[i]) * (1 / T_c)
        return ar + c * (_ct[i] - 1) * _ct[i] * (T_r ** (_ct[i] + 1))
    
    ar = fori_loop(0, 4, body1, ar)

    def body2(i, ar):
        ip = i + 4
        c = (
            _cr[ip]
            * (rho_r ** _crho[ip])
            * exp(-(rho_r ** _ce[i]))
            * (1 / T_c)
        )
        return ar + c * (_ct[ip] - 1) * _ct[ip] * (T_r ** (_ct[ip] + 1))

    ar = fori_loop(0, 18, body2, ar)

    return R_specific * ar

def d2residual_drhodT(rho, T):
    """
    Manually computed second mixed derivative of residual Helmholtz energy for performance critical situations
    """
    rho_r = rho / rho_c
    T_r = T_c / T

    ar = zeros_like(rho)
    def body1(i, ar):
        c = _cr[i] * (_crho[i] * rho_r ** (_crho[i] - 1)) * (1 / rho_c)
        return ar + c * (1 - _ct[i]) * (T_r ** _ct[i])
    
    ar = fori_loop(0, 4, body1, ar)

    def body2(i, ar):
        ip = i + 4
        c = (
            -_cr[ip]
            * (rho_r ** (_crho[ip] - 1))
            * (_ce[i] * rho_r ** _ce[i] - _crho[ip])
            * exp(-rho_r**_ce[i])
            * (1 / rho_c)
        )
        return ar + c * (1 - _ct[ip]) * (T_r ** _ct[ip])

    ar = fori_loop(0, 18, body2, ar)

    return R_specific * ar


''' Manual derivatives '''
_dAdrho                 = lambda rho, T: Jaeschke_Schley_drho(rho, T) + dresidual_drho(rho, T)
_dAdT                   = lambda rho, T: Jaeschke_Schley_dT(rho, T) + dresidual_dT(rho, T)
_d2AdrhodT              = lambda rho, T: Jaeschke_Schley_drhodT(rho, T) + d2residual_drhodT(rho, T)
_d2Adrho2               = lambda rho, T: Jaeschke_Schley_drho2(rho, T) + d2residual_drho2(rho, T)
_d2AdT2                 = lambda rho, T: Jaeschke_Schley_dT2(rho, T) + d2residual_dT2(rho, T)

''' Temperature equation (rho, p) -> T for initial conditions'''
_p                     = lambda rho, T: rho**2 * _dAdrho(rho, T)
root_func_pressure_T   = lambda T, rho, p: p - _p(rho, T)
drpdT_root             = lambda T, rho, p: - rho**2 * _d2AdrhodT(rho, T)

def temperature_rpt_Kunz_Wagner(rho, p, Tguess):
    """
        Solve temperature profile from density and pressure for Wagner gas
    """
    return solve_root_thermo(Tguess, rho, p, root_func_pressure_T, drpdT_root, 1e-10, 10)


''' Density equation (p, T) -> rho for initial conditions'''
root_func_pressure_rho = lambda rho, T, p: p - _p(rho, T)
drpdrho_root           = lambda rho, T, p: - 2 * rho * _dAdrho(rho, T) - rho**2 * _d2Adrho2(rho, T)


def density_ptr_Kunz_Wagner(p, T, rhoguess):
    """
        Solve density profile from pressure and temperature for Wagner gas
    """
    return solve_root_thermo(rhoguess, T, p, root_func_pressure_rho, drpdrho_root, 1e-10, 10)


''' Temperature equations (rho, e) -> T for simulations '''
_e                      = lambda rho, T: Kunz_Wagner(rho, T) - T * _dAdT(rho, T)
root_func_energy_T     = lambda T, rho, e: e - _e(rho, T)
dredT_root             = lambda T, rho, e: T * _d2AdT2(rho, T)

def temperature_ret_Kunz_Wagner(rho, e, Tguess):
    """
        Calculate temperature from density and specific internal energy for Wagner EOS.
    """
    return solve_root_thermo(Tguess, rho, e, root_func_energy_T, dredT_root, 1e-10, 10)

