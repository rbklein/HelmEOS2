"""
    Functions to compute the Gonzalez discrete gradient used in numerical fluxes
"""

from modules.thermodynamics.EOS import *
from modules.numerical.computation import zero_by_zero, lstsq2x2

def gonzalez_2vars(
        f : callable,
        dfdx : callable,
        dfdy : callable,
        x_1 : jnp.ndarray | float,
        x_2 : jnp.ndarray | float,
        y_1 : jnp.ndarray | float,
        y_2 : jnp.ndarray | float
) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[float, float]:
    """
    Compute the Gonzalez discrete gradient of a function f
    taking two variables, with a robust division operator
    for 0/0 = 0

    parameters:
    f       : 2 variable function
    dfdx    : analytical derivative of f wrt first var
    dfdy    : analytical derivative of f wrt second var
    x_1, x_2, y_1, y_2 : input variables

    returns:
    tuple containing two discrete gradient components
    """
    x_m = (x_1 + x_2) / 2
    y_m = (y_1 + y_2) / 2
    dx = x_2 - x_1
    dy = y_2 - y_1

    dx_f_m = dfdx(x_m, y_m)
    dy_f_m = dfdy(x_m, y_m)

    df = f(x_2, y_2) - f(x_1, y_1)

    factor_num = df - (dx_f_m * dx + dy_f_m * dy)
    factor_den = dx**2 + dy**2
    factor = zero_by_zero(factor_num, factor_den)

    dx_f = dx_f_m + factor * dx
    dy_f = dy_f_m + factor * dy

    return dx_f, dy_f












'''
OLD IMPLEMENTATION OF PCONS

def density_internal_energy_pressdg_gonzalez(rho_1, rho_2, T_1, T_2):
    """
    Gonzalez discrete gradient as used in the pressure_dg flux. 
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
    d_p_rho, d_p_eps = gonzalez_2vars(
        pressure_rho_eps,
        p_rho,
        p_eps,
        rho_1,
        rho_2,
        eps_1,
        eps_2
    )

    #p_rho discrete gradient 
    d_p_rho_rho, d_p_rho_eps = gonzalez_2vars(
        p_rho,
        p_rho_rho,
        p_rho_eps,
        rho_1,
        rho_2,
        eps_1,
        eps_2
    )

    #p_eps discrete gradient
    d_p_eps_rho, d_p_eps_eps = gonzalez_2vars(
        p_eps,
        p_eps_rho,
        p_eps_eps,
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

'''