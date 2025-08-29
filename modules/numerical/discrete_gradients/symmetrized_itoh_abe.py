"""
Implementation of the symmetrized Itoh-Abe discrete gradient

Hamiltonian-conserving discrete canonical equations based on variational difference quotients - T. Itoh, K. Abe
"""

from modules.thermodynamics.EOS import *

def symmetrized_itoh_abe_2vars(
        f : callable,
        dfdx : callable,
        dfdy : callable,
        d2fd2x : callable,
        d2fd2y : callable,
        #d3fd3x : callable,
        #d3fd3y : callable,
        x_1 : jnp.ndarray | float,
        x_2 : jnp.ndarray | float,
        y_1 : jnp.ndarray | float,
        y_2 : jnp.ndarray | float,
        tol : float = 1e-12
) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[float, float]:
    """
    Compute the symmetrized itoh-abe discrete gradient of a function f
    taking two variables

    parameters:
    f       : 2 variable function
    dfdx    : analytical derivative of f wrt first var
    dfdy    : analytical derivative of f wrt second var
    x_1, x_2, y_1, y_2 : input variables
    tol     : tolerance on x and y to use analytical derivatives

    returns:
    tuple containing two discrete gradient components
    """

    #compute function values
    f11 = f(x_1, y_1)
    f12 = f(x_1, y_2)
    f21 = f(x_2, y_1)
    f22 = f(x_2, y_2)

    #compute exact derivatives
    #dx_f_exact = 0.5 * (dfdx(x_1, y_1) + dfdx(x_2, y_2))
    #dy_f_exact = 0.5 * (dfdy(x_1, y_1) + dfdy(x_2, y_2))

    #compute derivatives for |x_2 - x_1| < eps range using taylor expansions (linear seems to also work)
    dx_f_taylor = dfdx(x_1, y_1) + 1/2 * d2fd2x(x_1,y_1) * (x_2 - x_1) #+ 1/27 * d3fd3x(x_1, y_1) * (x_2 - x_1)**2
    dy_f_taylor = dfdy(x_1, y_1) + 1/2 * d2fd2y(x_1,y_1) * (y_2 - y_1) #+ 1/27 * d3fd3y(x_1, y_1) * (y_2 - y_1)**2

    #compute discrete gradient components
    dx_f = jnp.where(
       jnp.isclose((x_2 - x_1), 0.0, atol = tol),
        dx_f_taylor,
        (0.5 * (f21 - f11) + 0.5 * (f22 - f12)) / (x_2 - x_1)
    )

    dy_f = jnp.where(
        jnp.isclose((y_2 - y_1), 0.0, atol = tol),
        dy_f_taylor,
        (0.5 * (f12 - f11) + 0.5 * (f22 - f21)) / (y_2 - y_1)
    )

    return dx_f, dy_f

