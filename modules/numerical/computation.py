"""
    modules with basic computation functions for numerical simulations.
"""

from prep_jax import *
from config.conf_geometry import *

from modules.geometry.grid  import GRID_SPACING, CELL_VOLUME
from jax.numpy              import finfo, sum, sqrt, maximum, cbrt
from jax                    import vmap

''' Derived parameters '''
EPS = finfo(DTYPE).eps

''' Auxiliary functions used in computation '''
def pad_1d_to_mesh(arr):
    return arr[(...,) + (None,) * (N_DIMENSIONS - 1)]

def extract_1d_from_padded(arr):
    idx = (slice(None),) + (0,) * (arr.ndim - 1)
    return arr[idx]

# def solve_root_thermo(v, vconst, vaux, root, droot, tol, it_max):
#     R = jnp.abs(vaux)

#     def cond(state):
#         v, i, not_done, status = state
#         return not_done
    
#     def body(state):
#         v, i, convd, status = state

#         res     = root(v, vconst, vaux)
#         dres    = droot(v, vconst, vaux)
#         step    = res / dres
#         v       = v - step

#         i           = i+1
#         status      = i < it_max
#         not_convd   = jnp.logical_not(jnp.all(jnp.abs(res) < tol * R))
#         not_done    = jnp.logical_and(status, not_convd) 

#         return v, i, not_done, status
    
#     sol, it, _, status = jax.lax.while_loop(cond, body, (v, 0, True, True))
#     return sol

def solve_root_thermo(v, vconst, vaux, root, droot, _1, _2):
    for _ in range(6):
        res     = root(v, vconst, vaux)
        dres    = droot(v, vconst, vaux)
        step    = res / dres
        v       = v - step
    return v

def vectorize_root(f : callable):
    f_vec = f
    for i in range(N_DIMENSIONS):
        f_vec = vmap(f_vec, (i,i,i), i)
    return f_vec

def spatial_average(field):
    integral = sum(field)
    for i in range(N_DIMENSIONS):
        integral *= GRID_SPACING[i] / DOMAIN_SIZE[i]
    return integral

def midpoint_integrate(field):
    field_scaled = field * CELL_VOLUME
    integral = sum(field_scaled)
    return integral

def zero_by_zero(num, den):
    """
    Robust division operator for 0/0 = 0 scenarios (thanks to Alessia)

    Does not work well in single precision
    """
    return den * (sqrt(2) * num) / (sqrt(den**4 + maximum(den, 100 * EPS)**4))

def cubic_root_single(coeffs):
    '''
    Cubic root solver that assumes a single real root
    '''
    a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * (a**2) * d) / (27 * a**3)
    delta = (q**2) / 4 + (p**3) / 27

    u = cbrt(-q/2 + sqrt(delta))
    v = cbrt(-q/2 - sqrt(delta))
    return u + v - b / (3 * a)
