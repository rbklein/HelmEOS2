"""
Implementation of the symmetrized Itoh-Abe discrete gradient

Hamiltonian-conserving discrete canonical equations based on variational difference quotients - T. Itoh, K. Abe
"""

from modules.thermodynamics.EOS import *

def symmetrized_itoh_abe_2vars(
        f : callable,
        dfdx : callable,
        dfdy : callable,
        x_1 : jnp.ndarray | float,
        x_2 : jnp.ndarray | float,
        y_1 : jnp.ndarray | float,
        y_2 : jnp.ndarray | float,
        atol : float = 1e-12,
        rtol : float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[float, float]:
    """
    Compute the symmetrized itoh-abe discrete gradient of a function f
    taking two variables

    The discrete gradient Gf between two points satisfies:
    (f2 - f1) = Gf^T (x2 - x1)

    parameters:
    f       : 2 variable function
    dfdx    : analytical derivative of f wrt first var
    dfdy    : analytical derivative of f wrt second var
    x_1, x_2, y_1, y_2 : input variables
    tol     : tolerance on x and y to use analytical derivatives

    returns:
    tuple containing two discrete gradient components
    """

    # Function values (4 corners)
    f11 = f(x_1, y_1)
    f12 = f(x_1, y_2)
    f21 = f(x_2, y_1)
    f22 = f(x_2, y_2)

    # Itoh-abe dg's at 0/0 
    dx_f_exact = 0.5 * (dfdx(x_1, y_1) + dfdx(x_2, y_2))
    dy_f_exact = 0.5 * (dfdy(x_1, y_1) + dfdy(x_2, y_2))

    # Differences and adaptive thresholds (abs + relative)
    dx = x_2 - x_1
    dy = y_2 - y_1

    scale_x = jnp.maximum(jnp.abs(x_1), jnp.abs(x_2))
    scale_y = jnp.maximum(jnp.abs(y_1), jnp.abs(y_2))

    thr_x = atol + rtol * scale_x
    thr_y = atol + rtol * scale_y

    use_diff_x = jnp.abs(dx) > thr_x
    use_diff_y = jnp.abs(dy) > thr_y

    # Symmetrized Itoh-abe numerators
    num_x = 0.5 * ((f21 - f11) + (f22 - f12))  
    num_y = 0.5 * ((f12 - f11) + (f22 - f21)) 

    # Safe denominators to avoid dividing by ~0 when masked out
    den_x = jnp.where(use_diff_x, dx, jnp.ones_like(dx))
    den_y = jnp.where(use_diff_y, dy, jnp.ones_like(dy))

    qx = num_x / den_x
    qy = num_y / den_y

    # Blend with 0/0 case where necessary
    dx_f = jnp.where(use_diff_x, qx, dx_f_exact)
    dy_f = jnp.where(use_diff_y, qy, dy_f_exact)

    return dx_f, dy_f

