"""
    Implementation of the 4th order Runge-Kutta method for solving ordinary differential equations (ODEs).
"""
from modules.numerical.flux import dudt as dudt_c
from modules.numerical.viscous import dudt as dudt_v
from modules.numerical.heat import dudt as dudt_q

def RK4(u, dt):
    """
    Perform one step of the 4th order Runge-Kutta method.

    Parameters:
    u (array-like): Current state of the system.
    dudt (function): Function that computes the derivative of u with respect to time.
    dt (float): Time step size.

    Returns:
    array-like: Updated state of the system after one time step.
    """
    k1 = dudt_c(u) + dudt_v(u) + dudt_q(u)
    u2 = u + 0.5 * dt * k1
    k2 = dudt_c(u2) + dudt_v(u2) + dudt_q(u2)
    u3 = u + 0.5 * dt * k2
    k3 = dudt_c(u3) + dudt_v(u3) + dudt_q(u3)
    u4 = u + dt * k3
    k4 = dudt_c(u4) + dudt_v(u4) + dudt_q(u4)
    
    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)