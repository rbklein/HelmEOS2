"""
Implementation of Wray's third order low-storage Runge-Kutta method
"""

from modules.numerical.flux     import dudt as dudt_c
from modules.numerical.viscous  import dudt as dudt_v
from modules.numerical.heat     import dudt as dudt_q
from modules.numerical.source   import dudt as dudt_s
from modules.thermodynamics.EOS import temperature


def _rhs(u, T, t):
    """Combined RHS: convective + viscous + heat."""
    rhs = dudt_c(u, T)
    rhs += dudt_v(u, T)
    rhs += dudt_q(u, T)

    return rhs

def Wray(u, T, dt, t):
    """
    One step of Wray's 3rd order low-storage Runge-Kutta method (2N scheme).

    see: "Minimal Storage Time-Advancement Schemes for Spectral Methods"
    """

    #compute
    r1 = u
    r2 = _rhs(u, T, t)

    #combine
    r1 = r1 + 1/4 * dt * r2
    r2 = (8/15 - 1/4) * dt * r2 + r1

    #compute
    #r1 = r1
    r2 = _rhs(r2, temperature(r2, T), t + 8/15 * dt)

    #combine
    #r1 = r1 + 0 * dt * r2
    r2 = (5/12 - 0) * dt * r2 + r1

    #compute
    #r1 = r1
    r2 = _rhs(r2, temperature(r2, T), t + (1/4 + 5/12) * dt)

    #combine
    r1 = r1 + 3/4 * dt * r2

    return r1