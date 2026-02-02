"""
    Implementation of the heat conduction coefficient as prescribed by Huber, Sykioti, Assael and Perkins (2016)

    ``Reference Correlation of the Thermal Conductivity of Carbon Dioxide from the Triple Point to 1100 K and up to 200 MPa''
"""

from prep_jax                   import *
from config.conf_thermodynamics import *

from modules.thermodynamics.EOS import pressure_rho, c_p, c_v
from jax.numpy                  import array, sqrt, ones_like, zeros_like, exp, atan
from jax.lax                    import fori_loop

''' Parameters '''

huber_zero_density_limit_coeffs = array([
    1.51874307e-2,
    2.80674040e-2,
    2.28564190e-2,
    -7.41624210e-3,
])

huber_residual_coeffs_const = array([
    1.00128e-2,
    5.60488e-2,
    -8.11620e-2,
    6.24337e-2,
    -2.06336e-2,
    2.53248e-3,
])

huber_residual_coeffs_temp = array([
    4.30829e-3,
    -3.58563e-2,
    6.71480e-2,
    -5.22855e-2,
    1.74571e-2,
    -1.96414e-3,
])

''' Functions '''

def check_consistency():
    """
    Consistency check
    """
    assert NAME_MOLECULE == "CO_2", "Huber thermal conductivity model is only valid for carbon dioxide" 

rho_c, T_c, p_c = molecule.critical_point

def huber_zero_density_limit(u, T):
    """
        Zero-density limit of the thermal conductivity in W m^-1 K^-1 according to Huber et al. (2016)
    """
    T_r = T / T_c
    num = sqrt(T_r)

    den = zeros_like(T)

    def zero_lim_body(i, ac):
        exponent = i
        return ac + huber_zero_density_limit_coeffs[i] / (T_r**exponent)

    den = fori_loop(0, 4, zero_lim_body, den)
    
    #correct units from mW to W
    return 1e-3 * num / den

def huber_residual_term(u, T):
    """
        Residual contribution to the thermal conductivity in W m^-1 K^-1 according too Huber et al. (2016)
    """
    rho_r = u[0] / rho_c
    T_r = T / T_c

    term = zeros_like(T)

    def residual_body(i, ac):
        exponent = i + 1
        return ac + (huber_residual_coeffs_const[i] + huber_residual_coeffs_temp[i] * T_r) * (rho_r**exponent)

    term = fori_loop(0, 6, residual_body, term)
    return term

from modules.thermodynamics.dynamic.laesecke_dynamic import laesecke_dynamic_viscosity

#from jax.debug import print as jax_print
#from jax.numpy import any, all, isnan

def huber_critical_enhancement(u, T):
    """
        Critical enhancement to the thermal conductivity in W m^-1 K^-1 according to Huber et al. (2016)
    """
    nu = 0.63
    R_D = 1.02
    gamma_small = 1.239
    gamma_capital = 0.052
    xi_0 = 1.5e-10
    q_D_inv = 4.0e-10
    T_ref = 456.19
    
    Cv = c_v(u[0], T)
    Cp = c_p(u[0], T)

    # jax_print("{rho}, {T}, {rhonan}, {Tnan}, {cv}, {cp}", 
    #           rho = all(u[0] > 0), 
    #           T = all(T > 0.0), 
    #           rhonan = any(isnan(u[0])), 
    #           Tnan = any(isnan(T)), 
    #           cv = any(isnan(Cv)), 
    #           cp = any(isnan(Cp))
    #     )

    xi_coeff = ((p_c * u[0]) / (gamma_capital * rho_c**2))**(nu / gamma_small)
    p_rho = pressure_rho(u[0], T)
    p_rho_ref = pressure_rho(u[0], T_ref * ones_like(T))
    xi = xi_0 * xi_coeff * (1 / p_rho - (T_ref / T) * 1 / p_rho_ref)**(nu / gamma_small)

    omega_exponent = - 1 / (q_D_inv / xi + (((xi * rho_c) / (q_D_inv * u[0]))**2) / 3)
    Omega_0 = 2 / PI * (1 - exp(omega_exponent))
    Omega = 2 / PI * ((Cp - Cv) / Cp * atan(xi / q_D_inv) + Cv / Cp * xi / q_D_inv)

    viscosity = laesecke_dynamic_viscosity(u, T)
    term = u[0] * Cp * R_D * BOLTZMANN_CONSTANT * T / (6 * PI * viscosity * xi) * (Omega - Omega_0)
    return term

def huber_thermal_conductivity(u, T):
    """
        Total thermal conductivity in W m^-1 K^-1 according to Huber et al. (2016)
    """
    #jax_print("{rho}", rho = any(T > 0))
    return huber_zero_density_limit(u, T) + huber_residual_term(u, T) + huber_critical_enhancement(u, T)

