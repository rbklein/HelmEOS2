"""
Implementation of the symmetrized Itoh-Abe discrete gradient

Hamiltonian-conserving discrete canonical equations based on variational difference quotients - T. Itoh, K. Abe
"""

from prep_jax import *

from modules.numerical.computation      import EPS
from jax.numpy                          import ndarray, sqrt, maximum, abs
from typing                             import Tuple

def symmetrized_itoh_abe_2vars(
        f : callable,
        dfdx : callable,
        dfdy : callable,
        x_1 : ndarray | float,
        x_2 : ndarray | float,
        y_1 : ndarray | float,
        y_2 : ndarray | float
) -> Tuple[ndarray, ndarray] | Tuple[float, float]:
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

        eps_dtype = EPS
        rtol = sqrt(eps_dtype)      # ~1e-8 for float64, ~3e-4 for float32
        atol = 10.0 * eps_dtype

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

        scale_x = maximum(abs(x_1), abs(x_2))
        scale_y = maximum(abs(y_1), abs(y_2))

        thr_x = atol + rtol * scale_x
        thr_y = atol + rtol * scale_y

        use_diff_x = abs(dx) > thr_x
        use_diff_y = abs(dy) > thr_y

        # Symmetrized Itoh-abe numerators
        num_x = 0.5 * ((f21 - f11) + (f22 - f12))  
        num_y = 0.5 * ((f12 - f11) + (f22 - f21)) 

        # Masks as floats (0.0 or 1.0)
        mx = use_diff_x.astype(f11.dtype)
        my = use_diff_y.astype(f11.dtype)

        dx_safe = dx * mx + (1.0 - mx)
        dy_safe = dy * my + (1.0 - my)

        qx_safe = num_x / dx_safe
        qy_safe = num_y / dy_safe

        # Branchless blend
        dx_f = dx_f_exact + mx * (qx_safe - dx_f_exact)
        dy_f = dy_f_exact + my * (qy_safe - dy_f_exact)
        
        return dx_f, dy_f

