"""
Taylor-Green vortex initial conditions
"""

from prep_jax               import *

from modules.geometry.grid      import DOMAIN_SIZE
from modules.thermodynamics.EOS import molecule, pressure, speed_of_sound
from jax.numpy                  import stack, ones_like, zeros_like, sin, cos, pi, array
from jax                        import jit
from modules.numerical.computation import pad_1d_to_mesh, extract_1d_from_padded

rho_c, T_c, p_c = molecule.critical_point

#@jit
def Taylor_Green_vortex_3d(mesh):
    """
        Domain : [- pi L, pi L]^3
        T      : ???
    """

    rho0    = pad_1d_to_mesh(array([1.1925 * rho_c]))
    T0      = pad_1d_to_mesh(array([1.1 * T_c]))

    # Determine p0, c0 from rho0 and T0
    p0      = pressure(rho0, T0)[0,0,0]
    c0      = speed_of_sound(rho0, T0)[0,0,0]
    U0      = 0.1 * c0
    
    rho0    = extract_1d_from_padded(rho0)[0]
    T0      = extract_1d_from_padded(T0)[0]

    L       = DOMAIN_SIZE[0] / (2 * pi)
    X, Y, Z = mesh[0] - pi * L, mesh[1] - pi * L, mesh[2] - pi * L

    p = p0 + rho0 * U0**2 / 16.0 * (cos(2 * X / L) + cos(2 * Y / L)) * (cos(2 * Z / L) + 2)
    T = T0 * ones_like(p)
    u = U0 * sin(X / L) * cos(Y / L) * sin(Z / L)
    v = - U0 * cos(X / L) * sin(Y / L) * sin(Z / L)
    w = zeros_like(u)

    return stack((u, v, w, p, T), axis = 0), 1 #vpt
