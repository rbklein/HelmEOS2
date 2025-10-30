"""
    Implementation of the forward Euler method for solving ordinary differential equations (ODEs).
"""
from modules.numerical.flux import dudt as dudt_c
from modules.numerical.viscous import dudt as dudt_v
from modules.numerical.heat import dudt as dudt_q

def forward_euler(u, T, dt):
    """
    Perform one step of the forward euler method.

    Parameters:
    u (array-like): Current state of the system.
    dudt (function): Function that computes the derivative of u with respect to time.
    dt (float): Time step size.

    Returns:
    array-like: Updated state of the system after one time step.
    """
    k = dudt_c(u, T) + dudt_v(u, T) + dudt_q(u, T)
    return u + dt * k