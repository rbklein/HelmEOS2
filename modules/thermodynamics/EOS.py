"""
Functions for thermodynamic equations of state (EOS).

This version is written to avoid common JAX/XLA memory blow-ups for real-fluid
Helmholtz EOS usage in flux functions (e.g., KEEP-style schemes).

Key changes vs. the original:
- Avoids taking grad() of functions that already call grad() (no "grad-of-grad"
  for Gibbs_beta / pressure_beta derivatives). Those derivatives are expressed
  analytically in terms of A_rho, A_T, A_rhorho, A_rhoT, A_TT.
- Computes 2nd derivatives using forward-over-reverse (jacfwd(grad)) rather than
  reverse-over-reverse (grad(grad)), which is typically much lighter on GPU memory.
- Keeps the public API largely compatible: pressure, entropy, internal_energy,
  Gibbs_energy, speed_of_sound, c_v, c_p, Gibbs_beta / pressure_beta and their
  derivatives, plus manufactured-solution derivatives.

Assumptions:
- Helmholtz_scalar(rho, T) returns specific Helmholtz free energy A per unit mass.
- All thermodynamic quantities below are per unit mass, except pressure which is per volume.
"""

from prep_jax import *
from config.conf_thermodynamics import *
from config.conf_geometry import N_DIMENSIONS

# -----------------------------
# Consistency checks
# -----------------------------
KNOWN_EOS = ["IDEAL_GAS", "VAN_DER_WAALS", "PENG_ROBINSON", "WAGNER"]
assert EOS in KNOWN_EOS, f"Unknown EOS: {EOS}"
assert MOLAR_MASS > 0, "Molar mass should be positive"

# -----------------------------
# Select EOS model
# -----------------------------
match EOS:
    case "IDEAL_GAS":
        from modules.thermodynamics.gas_models.ideal_gas import check_consistency_ideal_gas as check_consistency
        from modules.thermodynamics.gas_models.ideal_gas import ideal_gas as Helmholtz_scalar
        from modules.thermodynamics.gas_models.ideal_gas import temperature_rpt_ideal as temperature_rpt
        from modules.thermodynamics.gas_models.ideal_gas import density_ptr_ideal as density_ptr
        from modules.thermodynamics.gas_models.ideal_gas import temperature_ret_ideal as temperature_ret

    case "VAN_DER_WAALS":
        from modules.thermodynamics.gas_models.Van_der_Waals import check_consistency_Van_der_Waals as check_consistency
        from modules.thermodynamics.gas_models.Van_der_Waals import Van_der_Waals as Helmholtz_scalar
        from modules.thermodynamics.gas_models.Van_der_Waals import temperature_rpt_Van_der_Waals as temperature_rpt
        from modules.thermodynamics.gas_models.Van_der_Waals import density_ptr_Van_der_Waals as density_ptr
        from modules.thermodynamics.gas_models.Van_der_Waals import temperature_ret_Van_der_Waals as temperature_ret

    case "PENG_ROBINSON":
        from modules.thermodynamics.gas_models.Peng_Robinson import check_consistency_Peng_Robinson as check_consistency
        from modules.thermodynamics.gas_models.Peng_Robinson import Peng_Robinson as Helmholtz_scalar
        from modules.thermodynamics.gas_models.Peng_Robinson import temperature_rpt_Peng_Robinson as temperature_rpt
        from modules.thermodynamics.gas_models.Peng_Robinson import density_ptr_Peng_Robinson as density_ptr
        from modules.thermodynamics.gas_models.Peng_Robinson import temperature_ret_Peng_Robinson as temperature_ret

    case "WAGNER":
        from modules.thermodynamics.gas_models.Wagner import check_consistency_Wagner as check_consistency
        from modules.thermodynamics.gas_models.Wagner import Wagner as Helmholtz_scalar
        from modules.thermodynamics.gas_models.Wagner import temperature_rpt_Wagner as temperature_rpt
        from modules.thermodynamics.gas_models.Wagner import density_ptr_Wagner as density_ptr
        from modules.thermodynamics.gas_models.Wagner import temperature_ret_Wagner as temperature_ret

    case _:
        raise ValueError(f"Unknown EOS: {EOS}")

check_consistency()

# -----------------------------
# Temperature helper from conservative variables
# -----------------------------
def internal_energy_u(u):
    """
    Internal energy per unit mass from conservative state u = [rho, rho*u, rho*E].
    Works for N_DIMENSIONS velocity components.
    """
    rho = u[0]
    rhou = u[1 : N_DIMENSIONS + 1]
    rhoE = u[N_DIMENSIONS + 1]
    ke = 0.5 * jnp.sum(rhou * rhou, axis=0) / (rho * rho)
    return rhoE / rho - ke

# Temperature from conservative variables (u, Tguess) -> T
temperature = lambda u, Tguess: temperature_ret(u[0], internal_energy_u(u), Tguess)

# -----------------------------
# Vectorization helper (maps scalar thermo over N_DIMENSIONS array)
# -----------------------------
def _vectorize_thermo(f: callable) -> callable:
    """
    Vectorizes a scalar thermodynamic function to work elementwise over arrays
    of dim = N_DIMENSIONS (e.g., 3D mesh).
    """
    f_vec = jax.vmap(f, in_axes=(0, 0), out_axes=0)
    for i in range(1, N_DIMENSIONS):
        f_vec = jax.vmap(f_vec, in_axes=(i, i), out_axes=i)
    return f_vec

# -----------------------------
# Helmholtz derivatives (memory-aware)
# -----------------------------
# First derivatives: reverse-mode is fine and typically cheapest.
dAdrho_scalar = jax.grad(Helmholtz_scalar, argnums=0)
dAdT_scalar   = jax.grad(Helmholtz_scalar, argnums=1)

# Second derivatives: use forward-over-reverse for much lower peak memory vs grad(grad(.))
# (i.e., avoid reverse-over-reverse).
d2Ad2rho_scalar  = jax.jacfwd(dAdrho_scalar, argnums=0)
d2AdrhodT_scalar = jax.jacfwd(dAdrho_scalar, argnums=1)
d2Ad2T_scalar    = jax.jacfwd(dAdT_scalar,   argnums=1)

# Vectorized versions used on fields
Helmholtz = _vectorize_thermo(Helmholtz_scalar)
dAdrho    = _vectorize_thermo(dAdrho_scalar)
dAdT      = _vectorize_thermo(dAdT_scalar)

d2Ad2rho  = _vectorize_thermo(d2Ad2rho_scalar)
d2AdrhodT = _vectorize_thermo(d2AdrhodT_scalar)
d2Ad2T    = _vectorize_thermo(d2Ad2T_scalar)

# -----------------------------
# Core thermodynamic quantities
# -----------------------------
def pressure(rho, T):
    # p = rho^2 * A_rho
    return rho * rho * dAdrho(rho, T)

def entropy(rho, T):
    # s = -A_T
    return -dAdT(rho, T)

def internal_energy(rho, T):
    # e = A - T A_T
    return Helmholtz(rho, T) - T * dAdT(rho, T)

def Gibbs_energy(rho, T):
    # g = A + rho A_rho
    return Helmholtz(rho, T) + rho * dAdrho(rho, T)

def speed_of_sound(rho, T):
    """
    Speed of sound a (NOT squared).
    a^2 = dp/drho|s. For Helmholtz form:
    a^2 = 2 rho A_rho + rho^2 A_rhorho - rho^2 (A_rhoT^2)/A_TT
    """
    A_rho  = dAdrho(rho, T)
    A_rr   = d2Ad2rho(rho, T)
    A_rT   = d2AdrhodT(rho, T)
    A_TT   = d2Ad2T(rho, T)
    a2 = 2.0 * rho * A_rho + (rho * rho) * A_rr - (rho * rho) * (A_rT * A_rT) / A_TT
    return jnp.sqrt(jnp.abs(a2))

def c_v(rho, T):
    """
    Specific isochoric heat capacity c_v.
    c_v = -T * A_TT
    """
    return -T * d2Ad2T(rho, T)

def c_p(rho, T):
    """
    Specific isobaric heat capacity c_p.
    cp = cv + T * (dp/dT|rho)^2 / (rho^2 * dp/drho|T)
    where:
      dp/drho|T = 2 rho A_rho + rho^2 A_rhorho
      dp/dT|rho = rho^2 A_rhoT
    """
    A_rho  = dAdrho(rho, T)
    A_rr   = d2Ad2rho(rho, T)
    A_rT   = d2AdrhodT(rho, T)
    A_TT   = d2Ad2T(rho, T)

    cv = -T * A_TT
    dp_drho_T = 2.0 * rho * A_rho + (rho * rho) * A_rr
    dp_dT_rho = (rho * rho) * A_rT

    return cv + T * (dp_dT_rho * dp_dT_rho) / ((rho * rho) * dp_drho_T)

# -----------------------------
# Miscallaneous dynamic quantities
# -----------------------------
def kinetic_energy(rho, v):
    return 0.5 * rho * jnp.sum(v**2, axis = 0)

def total_energy(rho, T, v):
    return rho * internal_energy(rho, T) + kinetic_energy(rho, v)

# -----------------------------
# Quantities used in numerical fluxes (KEEP / beta-form)
# -----------------------------
# The original version used grad() on these beta-form functions which already
# called grad(Helmholtz), causing Hessian-level AD tapes and large intermediates.
# Here we compute them analytically from A and its 1st/2nd derivatives.
def _T_from_beta(beta):
    return 1.0 / beta

def Gibbs_beta_scalar(rho, beta):
    T = _T_from_beta(beta)
    A = Helmholtz_scalar(rho, T)
    A_rho = dAdrho_scalar(rho, T)
    return beta * (A + rho * A_rho)

def dgbdrho_scalar(rho, beta):
    T = _T_from_beta(beta)
    A_rho  = dAdrho_scalar(rho, T)
    A_rr   = d2Ad2rho_scalar(rho, T)
    # d/drho [beta*(A + rho*A_rho)] = beta*(A_rho + A_rho + rho*A_rr)
    return beta * (2.0 * A_rho + rho * A_rr)

def dgbdbeta_scalar(rho, beta):
    T = _T_from_beta(beta)
    A    = Helmholtz_scalar(rho, T)
    A_rho = dAdrho_scalar(rho, T)
    A_T   = dAdT_scalar(rho, T)
    A_rT  = d2AdrhodT_scalar(rho, T)

    g = A + rho * A_rho
    # d/dbeta [beta*g(rho,T)] with T=1/beta:
    # = g + beta * g_T * dT/dbeta, dT/dbeta=-1/beta^2 => = g - g_T/beta
    g_T = A_T + rho * A_rT
    return g - g_T / beta

def pressure_beta_scalar(rho, beta):
    T = _T_from_beta(beta)
    A_rho = dAdrho_scalar(rho, T)
    return beta * (rho * rho) * A_rho

def dpbdrho_scalar(rho, beta):
    T = _T_from_beta(beta)
    A_rho = dAdrho_scalar(rho, T)
    A_rr  = d2Ad2rho_scalar(rho, T)
    # d/drho [beta * rho^2 A_rho] = beta*(2 rho A_rho + rho^2 A_rr)
    return beta * (2.0 * rho * A_rho + (rho * rho) * A_rr)

def dpbdbeta_scalar(rho, beta):
    T = _T_from_beta(beta)
    A_rho = dAdrho_scalar(rho, T)
    A_rT  = d2AdrhodT_scalar(rho, T)
    # p_beta = beta * p, p = rho^2 A_rho, dp/dT = rho^2 A_rhoT
    # dp_beta/dbeta = p - (dp/dT)/beta
    return (rho * rho) * (A_rho - A_rT / beta)

# Vectorize beta-form functions over fields
Gibbs_beta   = _vectorize_thermo(Gibbs_beta_scalar)
dgbdrho      = _vectorize_thermo(dgbdrho_scalar)
dgbdbeta     = _vectorize_thermo(dgbdbeta_scalar)

pressure_beta = _vectorize_thermo(pressure_beta_scalar)
dpbdrho       = _vectorize_thermo(dpbdrho_scalar)
dpbdbeta      = _vectorize_thermo(dpbdbeta_scalar)

# -----------------------------
# Derivatives used in manufactured solutions (avoid grad; use analytic forms)
# -----------------------------
def pressure_rho_scalar(rho, T):
    A_rho = dAdrho_scalar(rho, T)
    A_rr  = d2Ad2rho_scalar(rho, T)
    return 2.0 * rho * A_rho + (rho * rho) * A_rr

def pressure_T_scalar(rho, T):
    A_rT = d2AdrhodT_scalar(rho, T)
    return (rho * rho) * A_rT

pressure_rho = _vectorize_thermo(pressure_rho_scalar)
pressure_T   = _vectorize_thermo(pressure_T_scalar)

def internal_energy_rho_scalar(rho, T):
    A_rho = dAdrho_scalar(rho, T)
    A_rT  = d2AdrhodT_scalar(rho, T)
    # e = A - T A_T => e_rho = A_rho - T A_rhoT
    return A_rho - T * A_rT

def internal_energy_T_scalar(rho, T):
    A_TT = d2Ad2T_scalar(rho, T)
    # e_T = -T * A_TT
    return -T * A_TT

internal_energy_rho = _vectorize_thermo(internal_energy_rho_scalar)
internal_energy_T   = _vectorize_thermo(internal_energy_T_scalar)

# -----------------------------
# Optional: Third derivatives
# -----------------------------
# If your code genuinely needs 3rd derivatives, prefer forward-mode chains as well.
# Leaving them out by default reduces the chance they are pulled into traces inadvertently.
# Uncomment if required.
#
# d3Ad3rho_scalar    = jax.jacfwd(d2Ad2rho_scalar,  argnums=0)
# d3Ad2rhodT_scalar  = jax.jacfwd(d2Ad2rho_scalar,  argnums=1)
# d3Ad2Tdrho_scalar  = jax.jacfwd(d2Ad2T_scalar,    argnums=0)
# d3Ad3T_scalar      = jax.jacfwd(d2Ad2T_scalar,    argnums=1)
#
# d3Ad3rho    = _vectorize_thermo(d3Ad3rho_scalar)
# d3Ad2rhodT  = _vectorize_thermo(d3Ad2rhodT_scalar)
# d3Ad2Tdrho  = _vectorize_thermo(d3Ad2Tdrho_scalar)
# d3Ad3T      = _vectorize_thermo(d3Ad3T_scalar)
