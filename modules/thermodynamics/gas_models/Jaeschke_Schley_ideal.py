"""
The Helmholtz energy and other functions of the ideal gas model of Jaeschke and Schley

- Used in the Kunz-Wagner high-accuracy model
- Specified for CO2 only
- Cannot be used directly as EOS, but is only used as helper in Kunz-Wagner EOS

see (referred to as KW-article): The GERG-2004 Wide-Range Equation of State for Natural Gases and Other Mixtures
and: Ideal-Gas Thermodynamic Properties for Natural-Gas Applications
"""

from prep_jax import *
from config.conf_thermodynamics import *

''' Derived parameters '''

R_specific = UNIVERSAL_GAS_CONSTANT / MOLAR_MASS #J K^-1 kg^-1 specific gas constant

''' set parameter values for Jaeschke-Schley '''

rho_c, T_c, p_c = molecule.critical_points

# parameters from Table A3.1 converted back to b_k form
_b0 = 2.500020000 + 1.0
_b13 = jnp.array([2.044520000, 2.033660000]) # sinh coeffs
_theta13 = jnp.array([3.022758166, 1.589964364]) * T_c 
_b24 = jnp.array([-1.060440000, 0.013930000]) # cosh coeffs
_theta24 = jnp.array([-2.844425476, 1.121596090]) * T_c

# reference values mentioned in KW-article
T_ref = 10 #298.15      # K
p_ref = 10 * R_specific #0.101325e6  # Pa
rho_ref = p_ref / (R_specific * T_ref) # kg m^-3 

coth = lambda x: 1.0 / jnp.tanh(x)


''' Helmholtz energy '''

def enthalpy_integral(T):
    '''
        Evaluates the anti-derivative of the c_p(T) model as in eq 4.14 of KW-article

        The quantity is multiplied by the specific gas constant to obtain per mass units
    '''
    linear_term = _b0 * T
    tanh_term = 0.0
    for i in range(2):
        tanh_term -= _b13[i] * _theta13[i] * jnp.tanh(_theta13[i] / T)
    coth_term = 0.0
    for i in range(2):
        coth_term += _b24[i] * _theta24[i] * coth(_theta24[i] / T) # coth = 1 / tanh
    return R_specific * (linear_term + tanh_term + coth_term) 

def entropy_integral(T):
    '''
        Evaluates the anti-derivative of c_p / T where c_p(T) is given by the model as in eq 4.14 of KW-article

        The quantity is multiplied by the specific gas constant to obtain per mass units
    '''
    ln_term = _b0 * jnp.log(T)
    sinh_term = 0.0
    for i in range(2):
        sinh_term -= _b13[i] * (jnp.log(jnp.sinh(_theta13[i] / T)) - _theta13[i] / T * coth(_theta13[i] / T))
    cosh_term = 0.0
    for i in range(2):
        cosh_term += _b24[i] * (jnp.log(jnp.cosh(_theta24[i] / T)) - _theta24[i] / T * jnp.tanh(_theta24[i] / T))
    return R_specific * (ln_term + sinh_term + cosh_term)

def Jaeschke_Schley(rho, T):
    '''
        Evaluate the ideal gas specific Helmholtz energy using the heat capacity model of Jaeschke and Schley in the KW-article

        Integrals are exactly evaluated using integral_calculator.com
    '''
    #enthalpy_terms  = enthalpy_integral(T) - enthalpy_integral(T_ref) # + 0.0 reference enthalpy
    #entropy_terms   = entropy_integral(T) - entropy_integral(T_ref) - R_specific * jnp.log(T / T_ref) - R_specific * jnp.log(rho / rho_ref) # + 0.0 reference entropy
    
    enthalpy_terms  = enthalpy_integral(T) - enthalpy_integral(T_ref) # + 0.0 reference enthalpy
    entropy_terms   = entropy_integral(T) - entropy_integral(T_ref) - R_specific * jnp.log(T / T_ref) - R_specific * jnp.log(rho / rho_ref) # + 0.0 reference entropy
    return enthalpy_terms - R_specific * T - T * entropy_terms