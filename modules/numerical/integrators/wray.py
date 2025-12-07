import jax.numpy as jnp

from modules.numerical.flux import dudt as dudt_c
from modules.numerical.viscous import dudt as dudt_v
from modules.numerical.heat import dudt as dudt_q
from modules.numerical.source import dudt as dudt_s
from modules.thermodynamics.EOS import temperature


def _rhs(u, T, t):
    """Combined RHS: convective + viscous + heat."""
    return (
        dudt_c(u, T)
        #+ dudt_v(u, T_est)
        #+ dudt_q(u, T_est)
        #+ dudt_s(u, T, t)
    )

def Wray(u, T, dt, t):
    """
    One step of Wray's 3rd order low-storage Runge-Kutta method (2N scheme).

    Low-storage form:

        r_0 = 0
        for s = 1..3:
            r_s = a_s * r_{s-1} + dt * f(u_{s-1})
            u_s = u_{s-1} + b_s * r_s

    Parameters
    ----------
    u : array-like
        Current state of the system (u^n).
    T : array-like
        Current temperature (consistent with u^n).
    dt : float
        Time step size.

    Returns
    -------
    array-like
        Updated state u^{n+1}.
    """

    # Wray / Williamson low-storage RK3 coefficients
    a1, a2, a3 = 0.0, -5.0 / 9.0, -153.0 / 128.0
    b1, b2, b3 = 1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0

    # Initialize low-storage arrays
    u_stage = u           # u_0
    T_stage = T           # T estimate corresponding to u_0
    r = jnp.zeros_like(u) # residual r_0

    # --- Stage 1 ---
    # Use the given T for the first stage (already consistent with u)
    r = a1 * r + dt * _rhs(u_stage, T_stage, t)
    u_stage = u_stage + b1 * r

    # --- Stage 2 ---
    # Update temperature estimate for new u_stage
    T_stage = temperature(u_stage, T_stage) 
    r = a2 * r + dt * _rhs(u_stage, T_stage, t + 8.0 * dt / 15.0)
    u_stage = u_stage + b2 * r

    # --- Stage 3 ---
    T_stage = temperature(u_stage, T_stage)
    r = a3 * r + dt * _rhs(u_stage, T_stage, t + 2.0 * dt / 3.0)
    u_stage = u_stage + b3 * r

    return u_stage
