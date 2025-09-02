"""
Implementation of mean for the pepc flux
"""

from prep_jax import *
from config.conf_numerical import *
from modules.thermodynamics.EOS import *
from modules.numerical.computation import lstsq2x2, EPS

# these functions can be combined to prevent recomputations

def density_internal_energy_pepc_mean_naive(rho_1, rho_2, T_1, T_2):
    """
    Pressure equilibrium preserving and pressure consistent means, naive implementation
    when two states are not close
    """

    # Pressures at the two states
    p_1 = pressure(rho_1, T_1)
    p_2 = pressure(rho_2, T_2)

    # Energy density eps = rho * e(rho, T)
    eps_1 = rho_1 * internal_energy(rho_1, T_1)
    eps_2 = rho_2 * internal_energy(rho_2, T_2)

    # Pressure derivatives
    pr1 = p_rho(rho_1, eps_1)   # p_ρ at state 1
    pr2 = p_rho(rho_2, eps_2)   # p_ρ at state 2
    pe1 = p_eps(rho_1, eps_1)   # p_ε at state 1
    pe2 = p_eps(rho_2, eps_2)   # p_ε at state 2

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

    # Arithmetic means
    rho_m = 0.5 * (rho_1 + rho_2)
    eps_m = 0.5 * (eps_1 + eps_2)

    density = rho_m + jnp.where(D != 0.0, num_rho / D, 0) # zero_by_zero(num_rho, D)  # num_rho / D  #    
    energy  = eps_m + jnp.where(D != 0.0, num_eps / D, 0) # zero_by_zero(num_eps, D)  # num_eps / D  #    

    return density, energy

def density_internal_energy_pepc_mean_robust(rho_1, rho_2, T_1, T_2):
    """
    Robust implementation of pepc means when states are close or equal
    """
    # Energy density eps = rho * e(rho, T)
    eps_1 = rho_1 * internal_energy(rho_1, T_1)
    eps_2 = rho_2 * internal_energy(rho_2, T_2)
    
    # Pressure values
    p_1 = pressure(rho_1, T_1)
    p_2 = pressure(rho_2, T_2)

    # Pressure derivatives
    pr1 = p_rho(rho_1, eps_2)
    pr2 = p_rho(rho_2, eps_2)
    pe1 = p_eps(rho_1, eps_2)
    pe2 = p_eps(rho_2, eps_2)

    dr = rho_2 - rho_1
    de = eps_2 - eps_1
    dp   = p_2 - p_1 
    dpr  = pr2 - pr1 
    dpe  = pe2 - pe1
    pr_m = 0.5 * (pr1 + pr2)
    pe_m = 0.5 * (pe1 + pe2)

    # residual chain rule 
    r_c = pr_m * dr + pe_m * de - dp

    # residual product/mean
    r_p = 1/4 * (dr * dpr + de * dpe)

    A = jnp.array([[dpr, dpe],
                   [pr_m, pe_m]])
    b = jnp.stack((r_c, r_p), axis = 0)
    sol = lstsq2x2(A, b)

    # Arithmetic means
    rho_m = 0.5 * (rho_1 + rho_2)
    eps_m = 0.5 * (eps_1 + eps_2)

    # Compute corrections
    density = rho_m + sol[0]
    energy = eps_m + sol[1]
    
    return density, energy


'''
This code is generated using ChatGPT 5
----------------------------------------------------------------------------------------------
'''
def density_internal_energy_pepc(rho_1, rho_2, T_1, T_2):
    """
    Pressure–equilibrium preserving / consistent (PEPC) mean for density and internal energy.

    Uses the *naive* closed-form by default, but **switches to the robust solver**
    for entries where the two states are nearly equal (or the naive denominator
    is ill-conditioned). All inputs are ndarrays defined on a computational mesh;
    selection is performed elementwise via JAX-friendly masking.

    Parameters
    ----------
    rho_1, rho_2 : array-like
        Densities of the left/right states.
    T_1, T_2     : array-like
        Temperatures of the left/right states.

    Returns
    -------
    density, energy : array-like
        PEPC means on the mesh, same shape as the inputs.
    """
    # First, compute the naive result everywhere
    dens_naive, ener_naive = density_internal_energy_pepc_mean_naive(rho_1, rho_2, T_1, T_2)

    # Build an elementwise mask where states are "nearly constant / equal"
    # or the naive formula becomes ill-conditioned.
    # We do this by checking small state deltas (relative) and a small Cramer's-rule denominator.
    # Avoid extra imports; use dtype-driven tolerances.
    eps_dtype = EPS
    rtol = jnp.sqrt(eps_dtype)      # ~1e-8 for float64, ~3e-4 for float32
    atol = 10.0 * eps_dtype

    # Basic thermodynamic quantities
    p_1 = pressure(rho_1, T_1)
    p_2 = pressure(rho_2, T_2)
    eps_1 = rho_1 * internal_energy(rho_1, T_1)
    eps_2 = rho_2 * internal_energy(rho_2, T_2)

    # Differences and scales
    dr = rho_2 - rho_1
    de = eps_2 - eps_1
    dp = p_2   - p_1

    rho_scale = (jnp.abs(rho_1) + jnp.abs(rho_2)) + 1.0
    eps_scale = (jnp.abs(eps_1) + jnp.abs(eps_2)) + 1.0
    p_scale   = (jnp.abs(p_1)   + jnp.abs(p_2))   + 1.0

    mask_close = (
        (jnp.abs(dr) <= atol + rtol * rho_scale) &
        (jnp.abs(de) <= atol + rtol * eps_scale) &
        (jnp.abs(dp) <= atol + rtol * p_scale)
    )

    # Ill-conditioning check via the naive denominator D
    pr1 = p_rho(rho_1, eps_1)
    pr2 = p_rho(rho_2, eps_2)
    pe1 = p_eps(rho_1, eps_1)
    pe2 = p_eps(rho_2, eps_2)

    dpr  = pr2 - pr1
    dpe  = pe2 - pe1
    pr_b = 0.5 * (pr1 + pr2)
    pe_b = 0.5 * (pe1 + pe2)

    D = dpr * pe_b - dpe * pr_b
    D_scale = jnp.abs(dpr * pe_b) + jnp.abs(dpe * pr_b) + 1.0
    mask_Dsmall = jnp.abs(D) <= (atol + rtol * D_scale)

    use_robust = mask_close | mask_Dsmall

    # Compute the robust result everywhere (JAX will JIT both branches efficiently)
    dens_robust, ener_robust = density_internal_energy_pepc_mean_robust(rho_1, rho_2, T_1, T_2)

    # Elementwise choose
    density = jnp.where(use_robust, dens_robust, dens_naive)
    energy  = jnp.where(use_robust, ener_robust, ener_naive)

    return density, energy
