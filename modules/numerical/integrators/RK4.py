"""
    Implementation of the 4th order Runge-Kutta method for solving ordinary differential equations (ODEs).
"""
from modules.numerical.flux import dudt as dudt_c
from modules.numerical.viscous import dudt as dudt_v
from modules.numerical.heat import dudt as dudt_q
from modules.thermodynamics.EOS import temperature

def RK4(u, T, dt):
    """
    Perform one step of the 4th order Runge-Kutta method.

    Parameters:
    u (array-like): Current state of the system.
    T (array-like): Current temperature of the system.
    dt (float): Time step size.

    Returns:
    array-like: Updated state of the system after one time step.
    """
    k1 = dudt_c(u, T) + dudt_v(u, T) + dudt_q(u, T)
    u2 = u + 0.5 * dt * k1
    T2 = temperature(u2, T)
    k2 = dudt_c(u2, T2) + dudt_v(u2, T2) + dudt_q(u2, T2)
    u3 = u + 0.5 * dt * k2
    T3 = temperature(u3, T)
    k3 = dudt_c(u3, T3) + dudt_v(u3, T3) + dudt_q(u3, T3)
    u4 = u + dt * k3
    T4 = temperature(u4, T)
    k4 = dudt_c(u4, T4) + dudt_v(u4, T4) + dudt_q(u4, T4)
    
    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)