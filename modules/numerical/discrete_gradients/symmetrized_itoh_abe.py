"""
Implementation of the symmetrized Itoh-Abe discrete gradient

Hamiltonian-conserving discrete canonical equations based on variational difference quotients - T. Itoh, K. Abe
"""

from modules.thermodynamics.EOS import *
from modules.numerical.computation import lstsq2x2

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

def density_internal_energy_keepdg_itoh_abe(rho_1, rho_2, T_1, T_2):
    """
    Symmetrized Itoh-Abe discrete gradient as used in the keep_dg flux. 
    The discrete gradient is computed with respect to inverted temperature (b or beta) and density.

    Parameters:
        - T_1 : jnp.ndarray, Temperature at side 1.
        - T_2 : jnp.ndarray, Temperature at side 2.
        - rho_1 : jnp.ndarray, Density at side 1.
        - rho_2 : jnp.ndarray, Density at side 2.
    """

    #gibbs energy discrete gradient
    d_gb_rho, d_gb_b = symmetrized_itoh_abe_2vars(
        Gibbs_beta, 
        dgbdrho,
        dgbdbeta,
        d2pbd2rho,
        d2pbd2beta,
        rho_1,
        rho_2,
        1/T_1,
        1/T_2
    )

    # pressure discrete gradient
    d_pb_rho, d_pb_b = symmetrized_itoh_abe_2vars(
        pressure_beta,
        dpbdrho,
        dpbdbeta,
        d2pbd2rho,
        d2pbd2beta,
        rho_1,
        rho_2,
        1/T_1,
        1/T_2
    )

    density = d_pb_rho / d_gb_rho
    internal_energy = d_gb_b - d_pb_b / density

    return density, internal_energy


def density_internal_energy_pressdg_itoh_abe(rho_1, rho_2, T_1, T_2):
    """
    Symmetrized Itoh-Abe discrete gradient as used in the pressure_dg flux. 
    The discrete gradient is computed with respect to density and the internal energy density.

    Parameters:
        - T_1 : jnp.ndarray, Temperature at side 1.
        - T_2 : jnp.ndarray, Temperature at side 2.
        - rho_1 : jnp.ndarray, Density at side 1.
        - rho_2 : jnp.ndarray, Density at side 2.
    """
    
    eps_1 = rho_1 * internal_energy(rho_1, T_1)
    eps_2 = rho_2 * internal_energy(rho_2, T_2)
    
    #p discrete gradient
    d_p_rho, d_p_eps = symmetrized_itoh_abe_2vars(
        pressure_rho_eps,
        p_rho,
        p_eps,
        p_rho_rho,
        p_eps_eps,
        rho_1,
        rho_2,
        eps_1,
        eps_2
    )

    #p_rho discrete gradient 
    d_p_rho_rho, d_p_rho_eps = symmetrized_itoh_abe_2vars(
        p_rho,
        p_rho_rho,
        p_rho_eps,
        p_rho_rho_rho,
        p_rho_eps_eps,
        rho_1,
        rho_2,
        eps_1,
        eps_2
    )

    #p_eps discrete gradient
    d_p_eps_rho, d_p_eps_eps = symmetrized_itoh_abe_2vars(
        p_eps,
        p_eps_rho,
        p_eps_eps,
        p_eps_rho_rho,
        p_eps_eps_eps,
        rho_1,
        rho_2,
        eps_1,
        eps_2
    )

    #compute arithmetic means
    rho_mean = 0.5 * (rho_1 + rho_2)
    eps_mean = 0.5 * (eps_1 + eps_2)
    p_rho_mean = 0.5 * (p_rho(rho_1, eps_1) + p_rho(rho_2, eps_2))
    p_eps_mean = 0.5 * (p_eps(rho_1, eps_1) + p_eps(rho_2, eps_2))

    #solve pressure consistent means in least-square sense
    discrete_hessian = jnp.stack([
        jnp.stack([d_p_rho_rho, d_p_rho_eps], axis = 0),
        jnp.stack([d_p_eps_rho, d_p_eps_eps], axis = 0)
    ], 
    axis = 1
    )

    grad_diff = jnp.stack([d_p_rho - p_rho_mean, d_p_eps - p_eps_mean], axis = 0)
    res = lstsq2x2(discrete_hessian, -grad_diff)
    
    density = rho_mean + res[0]
    energy = eps_mean + res[1]

    return density, energy


def density_internal_energy_pressdg_itoh_abe_ideal_gas(rho_1, rho_2, T_1, T_2):
    """
    Symmetrized Itoh-Abe discrete gradient as used in the pressure_dg flux. 
    The discrete gradient is computed with respect to density and the internal energy density.

    For the ideal gas case or other linear equations of state the discrete gradient systems reduces to arithmetic means

    Parameters:
        - T_1 : jnp.ndarray, Temperature at side 1.
        - T_2 : jnp.ndarray, Temperature at side 2.
        - rho_1 : jnp.ndarray, Density at side 1.
        - rho_2 : jnp.ndarray, Density at side 2.
    """
    
    eps_1 = rho_1 * internal_energy(rho_1, T_1)
    eps_2 = rho_2 * internal_energy(rho_2, T_2)

    #compute arithmetic means
    rho_mean = 0.5 * (rho_1 + rho_2)
    eps_mean = 0.5 * (eps_1 + eps_2)
    density = rho_mean 
    energy = eps_mean 

    return density, energy
