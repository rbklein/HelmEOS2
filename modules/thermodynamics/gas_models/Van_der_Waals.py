"""
The Helmholtz energy and other functions of the ideal gas law
"""

from prep_jax import *
from config.conf_thermodynamics import *

''' check parameter consistency '''

def check_consistency_Van_der_Waals():
    assert "a_VdW" in EOS_parameters, "EOS parameters for Van der Waals must include 'a_VdW'"
    assert "b_VdW" in EOS_parameters, "EOS parameters for Van der Waals must include 'b_VdW'"
    assert "molecular_dofs" in EOS_parameters, "EOS parameters for Van der Waals must include 'molecular_dofs'"
    assert isinstance(EOS_parameters["a_VdW"], (float, int)), "EOS parameter 'a_VdW' must be a number"
    assert isinstance(EOS_parameters["b_VdW"], (float, int)), "EOS parameter 'b_VdW' must be an integer"
    assert isinstance(EOS_parameters["molecular_dofs"], int), "EOS parameter 'molecular_dofs' must be a number"
    assert EOS_parameters["molecular_dofs"] > 0, "EOS parameter 'molecular_dofs' must be greater than 0"
    assert EOS_parameters["b_VdW"] > 0, "EOS parameter 'b_VdW' must be greater than 0"
    assert EOS_parameters["a_VdW"] > 0, "EOS parameter 'a_VdW' must be greater than 0"
    assert EOS_parameters["molecular_dofs"] >= 2, "EOS parameter 'molecular_dofs' must be at least 2 for Van der Waals EOS"

''' Derived parameters '''

R_specific = UNIVERSAL_GAS_CONSTANT / MOLAR_MASS #J K^-1 kg^-1 specific gas constant


''' set critical point values for Van der Waals '''

rho_c = 1 / (3 * EOS_parameters["b_VdW"])  # Critical density for Van der Waals
T_c = 8 / (R_specific * 27) * (EOS_parameters["a_VdW"] / EOS_parameters["b_VdW"])  # Critical temperature for Van der Waals
p_c = 1 / 27 * (EOS_parameters["a_VdW"] / (EOS_parameters["b_VdW"]**2))  # Critical pressure for Van der Waals

''' Helmholtz energy '''

def Van_der_Waals(rho, T):
    """
        Specific Helmholtz energy of the Van der Waals equation of state

        a_VdW = a_mol / M^2 (mass basis, M molar mass)
        b_VdW = b_mol / M

        critical point:
            - T_c   = 8/27 * a/b
            - rho_c = 1/3 * 1/b
            - p_c   = 1/27 * a/b^2 
    """
    a_VdW = EOS_parameters["a_VdW"]
    b_VdW = EOS_parameters["b_VdW"]
    molecular_dofs = EOS_parameters["molecular_dofs"]

    return - R_specific * T * (1 + jnp.log((1-rho*b_VdW) * T**(molecular_dofs / 2) / rho)) - a_VdW * rho


''' Temperature equation (rho, p) -> T for initial conditions'''

def temperature_eos_Van_der_Waals(rho, p):
    """
        Solve temperature profile from density and pressure for Van der Waals gas
    """
    a_VdW = EOS_parameters["a_VdW"]
    b_VdW = EOS_parameters["b_VdW"]
    T = (p + a_VdW * rho**2) * (1 / rho - b_VdW) / R_specific
    return T


''' Density equation (p, T) -> rho for initial conditions'''

from modules.numerical.computation import cubic_real_roots
def density_eos_Van_der_Waals(p, T):
    """
        Solve density profile from pressure and temperature for Van der Waals gas
    """
    a_VdW = EOS_parameters["a_VdW"]
    b_VdW = EOS_parameters["b_VdW"]

    #Van der Waals cubic polynomial coefficients
    p3 = - a_VdW * b_VdW * jnp.ones_like(p)
    p2 = a_VdW * jnp.ones_like(p)
    p1 = -(b_VdW * p + R_specific * T)
    p0 = p

    #solve cubic and take first real root (assumes only one real root)
    coeffs = jnp.stack((p3, p2, p1, p0), axis = 0)
    r, mask, num_real = cubic_real_roots(coeffs)
    rho = r[..., 0]
    return rho


''' Temperature equations (rho, e) -> T for simulations '''

def temperature_Van_der_Waals(rho, e):
    """
        Calculate temperature from density and specific internal energy for Van der Waals EOS.
    """
    a_VdW = EOS_parameters["a_VdW"]
    molecular_dofs = EOS_parameters["molecular_dofs"]
    T = (2 / (R_specific * molecular_dofs)) * (e + a_VdW * rho)
    return T