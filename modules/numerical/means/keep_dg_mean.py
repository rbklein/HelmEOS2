"""
Implementation of the density and internal energy means for the keep_dg flux
"""

from prep_jax import *
from config.conf_numerical import *
from modules.thermodynamics.EOS import *

''' Import necessary discrete gradients'''
match DISCRETE_GRADIENT:
    case "SYMMETRIZED_ITOH_ABE":
        from modules.numerical.discrete_gradients.symmetrized_itoh_abe import symmetrized_itoh_abe_2vars
    case "GONZALEZ":
        from modules.numerical.discrete_gradients.gonzalez import gonzalez_2vars
    case _:
        #handle not implemented as well
        raise ValueError(f"Unknown discrete gradient: {DISCRETE_GRADIENT}")
    

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


def density_internal_energy_keepdg_gonzalez(rho_1, rho_2, T_1, T_2):
    """
    Gonzalez discrete gradient as used in the keep_dg flux. 
    The discrete gradient is computed with respect to inverted temperature (b or beta) and density.

    Parameters:
        - rho_1 : jnp.ndarray, Density at side 1.
        - rho_2 : jnp.ndarray, Density at side 2.
        - T_1 : jnp.ndarray, Temperature at side 1.
        - T_2 : jnp.ndarray, Temperature at side 2.
    """

    #gibbs energy discrete gradient
    d_gb_rho, d_gb_b = gonzalez_2vars(
        Gibbs_beta, #lambda rho, beta: gibbs_energy(rho, 1/beta) * beta,
        dgbdrho,
        dgbdbeta,
        rho_1,
        rho_2,
        1/T_1,
        1/T_2
    )

    # pressure discrete gradient
    d_pb_rho, d_pb_b = gonzalez_2vars(
        pressure_beta, #lambda rho, beta: pressure(rho, 1/beta) * beta,
        dpbdrho,
        dpbdbeta,
        rho_1,
        rho_2,
        1/T_1,
        1/T_2
    )

    density = d_pb_rho / d_gb_rho
    internal_energy = d_gb_b - d_pb_b / density

    return density, internal_energy