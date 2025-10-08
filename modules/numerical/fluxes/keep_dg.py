"""
Implementation of a new kinetic energy consistent and entropy conserving numerical flux
for arbitrary equations of state.
"""

from prep_jax import *
from config.conf_numerical import *
from modules.geometry.grid import *
from modules.thermodynamics.EOS import *

''' Import necessary discrete gradient means'''
match DISCRETE_GRADIENT:
    case "SYM_ITOH_ABE":
        from modules.numerical.means.keep_dg_mean import density_internal_energy_keepdg_itoh_abe as density_internal_energy
    case "GONZALEZ":
        from modules.numerical.means.keep_dg_mean import density_internal_energy_keepdg_gonzalez as density_internal_energy
    case _:
        raise ValueError(f"Unknown discrete gradient: {DISCRETE_GRADIENT}")

def div_x_keepdg_2d(u):
    """
    Assume u is padded appropriately (5, n_x + 2, n_y + 2).
    """

    n_x, n_y = GRID_RESOLUTION
    d_x, d_y = GRID_SPACING

    #compute necessary fields
    density     = u[0] 
    temp        = temperature(u)
    vel         = u[1:3] / density
    press       = pressure(density, temp)

    density_mean, internal_energy_mean = density_internal_energy(
        density[:-1, :],
        density[1:, :],
        temp[:-1, :],
        temp[1:, :]
    )

    vel_mean    = 0.5 * (vel[:, :-1, :] + vel[:, 1:, :])
    f_density   = density_mean * vel_mean[0, :, :]

    #compute pressure in x-direction
    pressure_mean   = 0.5 * (press[:-1, :] + press[1:, :]) 
    f_p             = jnp.stack((pressure_mean, jnp.zeros_like(pressure_mean)), axis=0)
    f_momentum      = f_density[None, :, :] * vel_mean + f_p

    #compute ||v_m||^2 - 0.5 (||v||^2)_m
    kinetic_energy      = 0.5 * jnp.sum(vel**2, axis=0)
    kinetic_energy_mean = jnp.sum(vel_mean**2, axis=0) - 0.5 * (kinetic_energy[:-1, :] + kinetic_energy[1:, :])

    #compute 2 p_m v_m - (p v)_m
    pv              = press * vel[0, :, :]
    pv_mean         = 2 * pressure_mean * vel_mean[0, :, :] - 0.5 * (pv[:-1, :] + pv[1:, :])

    f_internal_energy   = f_density * internal_energy_mean
    f_kinetic_energy    = f_density * kinetic_energy_mean
    f_pv                = pv_mean    

    f_total_energy = f_internal_energy + f_kinetic_energy + f_pv
    f = jnp.stack((f_density, f_momentum[0, :, :], f_momentum[1, :, :], f_total_energy), axis=0)[:, :, 1:-1]

    return d_y * (f[:, 1:, :] - f[:, :-1, :])

def div_y_keepdg_2d(u):
    """
    Assume u is padded appropriately (5, n_x + 2, n_y + 2).
    """

    n_x, n_y = GRID_RESOLUTION
    d_x, d_y = GRID_SPACING

    #compute necessary fields
    density     = u[0] 
    temp        = temperature(u)
    vel         = u[1:3] / density
    press       = pressure(density, temp)

    density_mean, internal_energy_mean = density_internal_energy(
        density[:, :-1],
        density[:, 1:],
        temp[:, :-1],
        temp[:, 1:]
    )

    vel_mean    = 0.5 * (vel[:, :, :-1] + vel[:, :, 1:])
    f_density   = density_mean * vel_mean[1, :, :]

    #compute pressure in y-direction
    pressure_mean   = 0.5 * (press[:, :-1] + press[:, 1:]) 
    f_p             = jnp.stack((jnp.zeros_like(pressure_mean), pressure_mean), axis=0)
    f_momentum      = f_density[None, :, :] * vel_mean + f_p

    #compute ||v_m||^2 - 0.5 (||v||^2)_m
    kinetic_energy      = 0.5 * jnp.sum(vel**2, axis=0)
    kinetic_energy_mean = jnp.sum(vel_mean**2, axis=0) - 0.5 * (kinetic_energy[:, :-1] + kinetic_energy[:, 1:])

    #compute 2 p_m v_m - (p v)_m
    pv              = press * vel[1, :, :]
    pv_mean         = 2 * pressure_mean * vel_mean[1, :, :] - 0.5 * (pv[:, :-1] + pv[:, 1:])

    f_internal_energy   = f_density * internal_energy_mean
    f_kinetic_energy    = f_density * kinetic_energy_mean
    f_pv                = pv_mean    

    f_total_energy = f_internal_energy + f_kinetic_energy + f_pv
    f = jnp.stack((f_density, f_momentum[0, :, :], f_momentum[1, :, :], f_total_energy), axis=0)[:, 1:-1, :]

    return d_x * (f[:, :, 1:] - f[:, :, :-1])


def div_keep_dg_2d(u):
    return div_x_keepdg_2d(u) + div_y_keepdg_2d(u)

#placeholder for the actual implementation
def div_keep_dg_3d(u):
    pass

