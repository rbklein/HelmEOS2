"""
    Implementation of the forward Euler method for solving ordinary differential equations (ODEs).
"""
from modules.numerical.flux     import dudt as dudt_c
from modules.numerical.viscous  import dudt as dudt_v
from modules.numerical.heat     import dudt as dudt_q
from modules.numerical.source   import dudt as dudt_s


def _rhs(u, T, t):
    """Combined RHS: convective + viscous + heat."""
    return (
        dudt_c(u, T)
        + dudt_v(u, T)
        + dudt_q(u, T)
        + dudt_s(u, T, t)
    )

def forward_euler(u, T, dt, t):
    """
    Perform one step of the forward euler method.

    Parameters:
    u (array-like): Current state of the system.
    dudt (function): Function that computes the derivative of u with respect to time.
    dt (float): Time step size.
    t (float): current time

    Returns:
    array-like: Updated state of the system after one time step.
    """
    k = _rhs(u, T, t)
    return u + dt * k