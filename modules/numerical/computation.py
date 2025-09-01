"""
    modules with basic computation functions for numerical simulations.
"""

from prep_jax import *
from modules.geometry.grid import *

import numpy as np
from scipy.spatial import ConvexHull

''' Derived parameters '''
EPS = jnp.finfo(DTYPE).eps


''' Auxiliary functions used in computation '''
def spatial_average(field):
    integral = jnp.sum(field)
    for i in range(N_DIMENSIONS):
        integral *= GRID_SPACING[i] / DOMAIN_SIZE[i]
    return integral

def smoothed_jump(mesh_input, left_state, right_state, slope):
    return 0.5 * (left_state + right_state) + 0.5 * (right_state - left_state) * jnp.tanh(slope * mesh_input)

def zero_by_zero(num, den):
    """
    Robust division operator for 0/0 = 0 scenarios (thanks to Alessia)
    """
    return den * (jnp.sqrt(2) * num) / (jnp.sqrt(den**4 + jnp.maximum(den, 1e-14)**4))

def sign_nonzero(x):
    """
    Sign function that returns +1 for x >= 0, -1 otherwise.
    """
    return jnp.where(x >= 0, 1, -1) 

def rq2x2(A):
    """
    Compute the RQ-decomposition of a 2x2 matrix in every cell of the mesh
    analytically.

    Parameters:
    A : nd.array of shape (2, 2, nx, ny(, nz)) containing 2x2 matrices

    Returns:
    x, y, z, c2, s2 : nd.arrays of shape (nx, ny(, nz)) containing R,Q components

    Implementation edited from C++ to JAX:
    https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
    """
    a, b = A[0,0, ...], A[0,1, ...]
    c, d = A[1,0, ...], A[1,1, ...]

    maxden = jnp.maximum(jnp.abs(c), jnp.abs(d))
    rcmaxden = 1.0 / maxden
    c_scaled = c * rcmaxden
    d_scaled = d * rcmaxden

    den = 1.0 / jnp.sqrt(c_scaled*c_scaled + d_scaled*d_scaled)

    numx = -b*c_scaled + a*d_scaled
    numy =  a*c_scaled + b*d_scaled

    x = numx * den
    y = numy * den
    z = maxden / den

    s2 = -c_scaled * den
    c2 =  d_scaled * den

    cond = (c == 0.0)
    x = jnp.where(cond, a, x)
    y = jnp.where(cond, b, y)
    z = jnp.where(cond, d, z)
    c2 = jnp.where(cond, 1.0, c2)
    s2 = jnp.where(cond, 0.0, s2)

    return x, y, z, c2, s2

def svd2x2(A):
    """
    Compute the SVD of a 2x2 matrix in every cell of the mesh
    analytically.

    The singular vectors and values are NOT ordered according
    to singular value size

    Parameters:
    A : nd.array of shape (2, 2, nx, ny(, nz)) containing 2x2 matrices

    Returns:
    u1, u2, v1, v2: nd.arrays of shape (2, nx, ny(, nz)) containing singular vectors
    d1, d2 : nd.arrays of shape (nx, ny(, nz)) containing singular values

    Implementation edited from C++ to JAX:
    https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
    """
    x, y, z, c2, s2 = rq2x2(A)

    mxy = jnp.maximum(jnp.abs(x), jnp.abs(y))

    scaler = jnp.where(mxy != 0.0, 1 / mxy, 1.0) 
    x_ = x * scaler
    y_ = y * scaler
    z_ = z * scaler

    numer = (z_-x_)*(z_+x_) + y_*y_
    gamma = jnp.where(numer == 0, 1.0, x_*y_)

    cond = (gamma != 0)
    zeta = jnp.where(cond, numer / gamma, 0)
    t = jnp.where(
            cond,
            2.0 * sign_nonzero(zeta) / (jnp.abs(zeta) + jnp.sqrt(zeta*zeta + 4.0)),
            0 
        )
    
    c1 = 1.0 / jnp.sqrt(1.0 + t*t)
    s1 = c1 * t

    usa =  c1*x - s1*y
    usb =  s1*x + c1*y
    usc = -s1*z
    usd =  c1*z

    t   = c1*c2 + s1*s2
    s2p = c2*s1 - c1*s2
    c2p = t
    c2, s2 = c2p, s2p

    d1 = jnp.hypot(usa, usc)
    d2 = jnp.hypot(usb, usd)

    cond = (d2 > d1)
    dmax = jnp.where(cond, d2, d1)
    usmax1 = jnp.where(cond, usd, usa)
    usmax2 = jnp.where(cond, usb, -usc)
    signd1 = sign_nonzero(x * z)
    dmax *= jnp.where(cond, signd1, 1)
    d2 *= signd1

    cond = (dmax != 0.0)
    rcpd = jnp.where(cond, 1/dmax, 0)
    c1 = jnp.where(cond, usmax1 * rcpd, 1)
    s1 = usmax2 * rcpd

    u1 = jnp.stack((c1, -s1), axis = 0)
    u2 = jnp.stack((s1, c1), axis = 0)
    
    sd1 = sign_nonzero(d1)
    sd2 = sign_nonzero(d2)
    
    v1 = jnp.stack((sd1 * c2, sd1 * -s2), axis = 0)
    v2 = jnp.stack((sd2 * s2, sd2 * c2), axis = 0)

    d1 = jnp.abs(d1)
    d2 = jnp.abs(d2)

    return u1, u2, d1, d2, v1, v2

def lstsq2x2(A, b):
    """
    Solve a 2x2 least squares problem in every cell of the mesh
    Returns the least norm solution in cases of nonuniqueness

    Parameters:
    A : nd.array of shape (2, 2, nx, ny(, nz)) containing 2x2 matrices
    b : nd.array of shape (2, nx, ny(, nz)) containing the rhs
    tol : float cutoff for small singular values, multiplies machine precision

    Returns:
    nd.array of shape (2, nx, ny(, nz)) containing the solutions
    """
    u1, u2, d1, d2, v1, v2 = svd2x2(A)

    # project b onto each left singular vector
    alpha1 = u1[0]*b[0] + u1[1]*b[1]   # shape (*batch)
    alpha2 = u2[0]*b[0] + u2[1]*b[1]

    dmax = jnp.maximum(jnp.abs(d1), jnp.abs(d2))
    rcond = EPS * 2.0

    # form safe reciprocals of singular values
    inv_d1 = jnp.where(jnp.abs(d1) > (dmax * EPS), 1.0/d1, 0.0)
    inv_d2 = jnp.where(jnp.abs(d2) > (dmax * EPS), 1.0/d2, 0.0)

    # coefficients in V-basis
    gamma1 = inv_d1 * alpha1           # (*batch)
    gamma2 = inv_d2 * alpha2

    # reconstruct x = gamma1·v1 + gamma2·v2
    x0 = gamma1 * v1[0] + gamma2 * v2[0]
    x1 = gamma1 * v1[1] + gamma2 * v2[1]

    return jnp.stack((x0, x1), axis=0)

def convex_envelope(x, fs):
    """
        Compute indices of the lower convex envelope of a function, code adapted from:

            "https://gist.github.com/parsiad/56a68c96bd3d300cb92f0c03c68198ee"
    """
    N = fs.shape[0]
    
    fs_pad = np.empty(N+2)
    fs_pad[1:-1], fs_pad[0], fs_pad[-1] = fs, np.max(fs) + 1.0, np.max(fs) + 1.0
    
    x_pad = np.empty(N+2)
    x_pad[1:-1], x_pad[0], x_pad[-1] = x, x[0], x[-1]
    
    epi = np.column_stack((x_pad, fs_pad))
    hull = ConvexHull(epi)
    result = [v-1 for v in hull.vertices if 0 < v <= N]
    result.sort()
    
    return jnp.array(result, dtype=jnp.int32)

'''
This code is generated using ChatGPT 5
----------------------------------------------------------------------------------------------
'''
def _cbrt(z):
    """Principal complex cube root, vectorized and JIT-friendly."""
    z = z + 0j  # ensure complex dtype
    r = jnp.abs(z)
    th = jnp.angle(z)
    return jnp.where(r == 0, 0.0 + 0.0j, (r ** (1.0/3.0)) * jnp.exp(1j * th / 3.0))


def _realify(z, tol=1e-12):
    """Convert nearly-real complex values to real scalars."""
    real_part = jnp.real(z)
    imag_part = jnp.imag(z)
    mask = jnp.abs(imag_part) < tol * (1 + jnp.abs(real_part))
    return jnp.where(mask, real_part, z)


def cubic_roots(coeffs, tol=1e-12):
    """
    Vectorized cubic polynomial root solver using Cardano's method.

    Args:
        coeffs: array of shape (4, ...), first axis [a, b, c, d].
                Represents a*x^3 + b*x^2 + c*x + d = 0.
        tol: tolerance for considering roots real.

    Returns:
        array of shape (3, ...) with roots. Roots that are numerically real
        within `tol` are returned as floats.
    """
    coeffs = jnp.asarray(coeffs)
    a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

    # Normalize to monic
    b = b / a
    c = c / a
    d = d / a

    # Depressed cubic substitution: x = y - b/3
    b_c = b + 0j
    p = c - (b**2) / 3.0
    q = (2.0 * b**3) / 27.0 - (b * c) / 3.0 + d
    p_c = p + 0j
    q_c = q + 0j

    delta = (q_c / 2.0) ** 2 + (p_c / 3.0) ** 3
    sqrt_delta = jnp.sqrt(delta)

    u = _cbrt(-q_c / 2.0 + sqrt_delta)
    v = _cbrt(-q_c / 2.0 - sqrt_delta)

    uv_target = -p_c / 3.0
    u_nonzero = jnp.abs(u) > 1e-30
    v = jnp.where(u_nonzero, uv_target / u, v)

    omega = -0.5 + 0.5j * jnp.sqrt(3.0)
    omega2 = jnp.conj(omega)

    y1 = u + v
    y2 = u * omega + v * omega2
    y3 = u * omega2 + v * omega

    roots = jnp.stack([y1 - b_c / 3.0, y2 - b_c / 3.0, y3 - b_c / 3.0], axis=-1)

    # Realify nearly-real roots
    roots = _realify(roots, tol=tol)
    return roots


def _is_real(z, tol=1e-12):
    """Scale-aware test for (numerically) real complex values.
    Returns a boolean mask with True where Im(z) is negligible.
    """
    return jnp.abs(jnp.imag(z)) <= tol * (1.0 + jnp.abs(jnp.real(z)))


def cubic_roots_coerced(coeffs, tol=1e-12):
    """Like cubic_roots, but zeroes tiny imaginary parts on real roots.

    Returns:
        complex array with same shape as cubic_roots(...), but entries that
        are numerically real have exactly zero imaginary component.
    """
    z = cubic_roots(coeffs)
    mask = _is_real(z, tol)
    return jnp.where(mask, jnp.real(z) + 0.0j, z)


def cubic_real_roots(coeffs, tol=1e-12, sort=True):
    """
    Extract real roots from cubic polynomials.

    Args:
        coeffs: array of shape (4, ...) with [a,b,c,d].
        tol: numerical tolerance for deciding if a root is real.
        sort: if True, sort real roots ascending and place NaNs last.

    Returns:
        roots_real: float array of shape (3, ...) with real roots; NaN where not real.
        is_real_mask: boolean array of shape (3, ...) indicating which entries are real.
        num_real: int array of shape (...) counting real roots per polynomial.
    """
    z = cubic_roots(coeffs)
    mask = _is_real(z, tol)
    roots_real = jnp.where(mask, jnp.real(z), jnp.nan)

    if sort:
        # Sort along last axis, pushing NaNs to the end
        keys = jnp.where(jnp.isnan(roots_real), jnp.inf, roots_real)
        idx = jnp.argsort(keys, axis=-1)
        roots_real = jnp.take_along_axis(roots_real, idx, axis=-1)
        mask = jnp.take_along_axis(mask, idx, axis=-1)

    num_real = jnp.sum(mask, axis=-1)
    return roots_real, mask, num_real

''' 
----------------------------------------------------------------------------------------------
'''