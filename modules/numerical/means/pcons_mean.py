"""
Implementation of mean for the pcons flux
"""

from prep_jax import *
from config.conf_numerical import *
from modules.thermodynamics.EOS import *
from modules.numerical.computation import zero_by_zero


def density_internal_energy_pcons_gonzalez_mean(rho_1, rho_2, T_1, T_2):
    """
    A Gonzalez-like density and internal density energy mean that minimizes the difference with
    the arithmetic mean in a reduced thermodynamic norm while satisfying the PPE constraint

    For an injective map (rho, eps) -> (p_rho, p_eps) the method is well-posed with a robust numerical
    evaluation

    Parameters:
        - T_1 : jnp.ndarray, Temperature at side 1.
        - T_2 : jnp.ndarray, Temperature at side 2.
        - rho_1 : jnp.ndarray, Density at side 1.
        - rho_2 : jnp.ndarray, Density at side 2.
    """
    p_1 = pressure(rho_1, T_1)
    p_2 = pressure(rho_2, T_2)

    eps_1 = rho_1 * internal_energy(rho_1, T_1)
    eps_2 = rho_2 * internal_energy(rho_2, T_2)

    p_rho_1 = p_rho(rho_1, eps_1)
    p_rho_2 = p_rho(rho_2, eps_2)

    p_eps_1 = p_eps(rho_1, eps_1)
    p_eps_2 = p_eps(rho_2, eps_2)

    combined_diff = (rho_2 * p_rho_2 + eps_2 * p_eps_2 - p_2) - (rho_1 * p_rho_1 + eps_1 * p_eps_1 - p_1)
    diff_p_rho = p_rho_2 - p_rho_1
    diff_p_eps = p_eps_2 - p_eps_1

    rho_m = (rho_1 + rho_2) / 2.0
    eps_m = (eps_1 + eps_2) / 2.0

    num = combined_diff - (rho_m * diff_p_rho + eps_m * diff_p_eps)
    den = eps_c**2 * diff_p_eps**2 + rho_c**2 * diff_p_rho**2 #norm D = diag(rho_c^-2, eps_c^-2), den is multiplied by D^-1

    coeff = zero_by_zero(num, den)
    
    density = rho_m + coeff * rho_c**2 * diff_p_rho
    energy = eps_m + coeff * eps_c**2 * diff_p_eps

    return density, energy