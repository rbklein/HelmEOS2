"""
The Helmholtz energy and other functions of the Van der Waals equation
"""

from prep_jax import *
from config.conf_thermodynamics import *
from modules.numerical.computation import cubic_root_single

''' check parameter consistency '''

def check_consistency_Van_der_Waals():
    assert "molecular_dofs" in EOS_parameters, "EOS parameters for Van der Waals must include 'molecular_dofs'"
    assert isinstance(EOS_parameters["molecular_dofs"], int), "EOS parameter 'molecular_dofs' must be a number"
    assert EOS_parameters["molecular_dofs"] > 0, "EOS parameter 'molecular_dofs' must be greater than 0"
    assert EOS_parameters["molecular_dofs"] >= 2, "EOS parameter 'molecular_dofs' must be at least 2 for Van der Waals EOS"

''' Derived parameters '''


''' set critical point values for Van der Waals '''

# rho_c   = 1 / (3 * EOS_parameters["b_VdW"])                                             # Critical density for Van der Waals
# T_c     = 8 / (R_specific * 27) * (EOS_parameters["a_VdW"] / EOS_parameters["b_VdW"])   # Critical temperature for Van der Waals
# p_c     = 1 / 27 * (EOS_parameters["a_VdW"] / (EOS_parameters["b_VdW"]**2))             # Critical pressure for Van der Waals

rho_c, T_c, p_c = molecule.critical_point

a_VdW = 27.0 / 64.0 * R_specific**2 * T_c**2 / p_c
b_VdW = 1.0 / 8.0 * R_specific * T_c / p_c 

''' Helmholtz energy '''

def Van_der_Waals(rho, T):
    """
        Specific Helmholtz energy of the Van der Waals equation of state

        a_VdW = a_mol / M^2 (mass basis, M molar mass)
        b_VdW = b_mol / M

        critical point:
            - T_c   = 8/27 * a/(R*b)
            - rho_c = 1/3 * 1/b
            - p_c   = 1/27 * a/b^2 
    """
    molecular_dofs = EOS_parameters["molecular_dofs"]

    return - R_specific * T * (1 + jnp.log((1-rho*b_VdW) * T**(molecular_dofs / 2) / rho)) - a_VdW * rho


''' Temperature equation (rho, p) -> T for initial conditions'''
def temperature_rpt_Van_der_Waals(rho, p, Tguess):
    """
        Solve temperature profile from density and pressure for Van der Waals gas
    """
    T = (p + a_VdW * rho**2) * (1 / rho - b_VdW) / R_specific
    return T


''' Density equation (p, T) -> rho for initial conditions'''
def density_ptr_Van_der_Waals(p, T, rhoguess):
    """
        Solve density profile from pressure and temperature for Van der Waals gas
    """

    #Van der Waals cubic polynomial coefficients
    p3 = - a_VdW * b_VdW * jnp.ones_like(p)
    p2 = a_VdW * jnp.ones_like(p)
    p1 = -(b_VdW * p + R_specific * T)
    p0 = p

    #solve cubic and take first real root (assumes only one real root)
    coeffs = jnp.stack((p3, p2, p1, p0), axis = 0)
    rho = cubic_root_single(coeffs)
    return rho


''' Temperature equations (rho, e) -> T for simulations '''

def temperature_ret_Van_der_Waals(rho, e, Tguess):
    """
        Calculate temperature from density and specific internal energy for Van der Waals EOS.
    """
    molecular_dofs = EOS_parameters["molecular_dofs"]
    T = (2 / (R_specific * molecular_dofs)) * (e + a_VdW * rho)
    return T