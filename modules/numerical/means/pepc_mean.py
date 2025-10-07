"""
Implementation of mean for the pepc flux
"""

from prep_jax import *
from config.conf_numerical import *
from modules.thermodynamics.EOS import *
from modules.numerical.computation import EPS, smooth_indicator

def density_internal_energy_pepc(rho_1, rho_2, T_1, T_2):
    """
    Conditionally pressure equilibrium preserving and pressure consistent means 
    for density and internal energy
    """
    
    # Pressures at the two states
    p_1 = pressure(rho_1, T_1)
    p_2 = pressure(rho_2, T_2)

    # Energy density eps = rho * e(rho, T)
    eps_1 = rho_1 * internal_energy(rho_1, T_1)
    eps_2 = rho_2 * internal_energy(rho_2, T_2)

    # Arithmetic means
    rho_m = 0.5 * (rho_1 + rho_2)
    eps_m = 0.5 * (eps_1 + eps_2)

    # Pressure derivatives
    pr1 = p_rho(rho_1, eps_1)   # p_ρ at state 1
    pr2 = p_rho(rho_2, eps_2)   # p_ρ at state 2
    pe1 = p_eps(rho_1, eps_1)   # p_eps at state 1
    pe2 = p_eps(rho_2, eps_2)   # p_eps at state 2

    # Differences and averages
    dr   = rho_2 - rho_1
    de   = eps_2 - eps_1 
    dp   = p_2 - p_1
    dpr  = pr2 - pr1
    dpe  = pe2 - pe1
    pr_b = 0.5 * (pr1 + pr2)
    pe_b = 0.5 * (pe1 + pe2)

    # residual chain rule 
    r_c = pr_b * dr + pe_b * de - dp

    # residual product/mean
    r_p = 1/4 * (dr * dpr + de * dpe)

    # Denominator of Cramer's rule
    D = dpr * pe_b - dpe * pr_b

    # Numerators
    num_rho = pe_b * r_c - dpe * r_p
    num_eps = dpr * r_p - pr_b * r_c

    # Check degeneracy / ill-conditioning
    R = 2 * EPS * (jnp.abs(pr_b)**2 + jnp.abs(pe_b)**2 + jnp.abs(dpr)**2 + jnp.abs(dpe)**2) + 1e-10
    use_robust = jnp.abs(D) < R

    density = rho_m + jnp.where(
        use_robust,
        0,
        smooth_indicator(D, R, 0.1 * R) * (num_rho / D) #smooth transition to pepc corrected flux
    )

    energy = eps_m + jnp.where(
        use_robust,
        0,
        smooth_indicator(D, R, 0.1 * R) * (num_eps / D)
    )

    return density, energy
