"""
Functions to compute a naive heat flux for testing purposes
"""

from prep_jax import *
from config.conf_numerical import *
from modules.geometry.grid import *
from modules.thermodynamics.EOS import *
from modules.thermodynamics.constitutive import *

''' 3D versions of naive heat flux divergence '''

def div_x_naive_heat_flux_3d(u, T):
    '''
    Assume u is padded appropriately (5, n_x + 2, n_y + 2, n_z + 2)
    '''

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    k = thermal_conductivity(u, T)
    k_m = 0.5 * (k[1:, 1:-1, 1:-1] + k[:-1, 1:-1, 1:-1])

    dT_dx = (T[1:, 1:-1, 1:-1] - T[:-1, 1:-1, 1:-1]) / d_x
    q = k_m * dT_dx

    f_rho_x = jnp.zeros_like(q)
    f_m1_x = jnp.zeros_like(q)
    f_m2_x = jnp.zeros_like(q)
    f_m3_x = jnp.zeros_like(q)
    f_E_x = q

    F = jnp.stack((f_rho_x, f_m1_x, f_m2_x, f_m3_x, f_E_x), axis = 0)
    return d_y * d_z * (F[:, 1:, :, :] - F[:, :-1, :, :])

def div_y_naive_heat_flux_3d(u, T):
    '''
    Assume u is padded appropriately (5, n_x + 2, n_y + 2, n_z + 2)
    '''

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    k = thermal_conductivity(u, T)
    k_m = 0.5 * (k[1:-1, 1:, 1:-1] + k[1:-1, :-1, 1:-1])

    dT_dy = (T[1:-1, 1:, 1:-1] - T[1:-1, :-1, 1:-1]) / d_y
    q = k_m * dT_dy

    f_rho_y = jnp.zeros_like(q)
    f_m1_y = jnp.zeros_like(q)
    f_m2_y = jnp.zeros_like(q)
    f_m3_y = jnp.zeros_like(q)
    f_E_y = q

    F = jnp.stack((f_rho_y, f_m1_y, f_m2_y, f_m3_y, f_E_y), axis = 0)
    return d_x * d_z * (F[:, :, 1:, :] - F[:, :, :-1, :])

def div_z_naive_heat_flux_3d(u, T):
    '''
    Assume u is padded appropriately (5, n_x + 2, n_y + 2, n_z + 2)
    '''

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    k = thermal_conductivity(u, T)
    k_m = 0.5 * (k[1:-1, 1:-1, 1:] + k[1:-1, 1:-1, :-1])

    dT_dz = (T[1:-1, 1:-1, 1:] - T[1:-1, 1:-1, :-1]) / d_z
    q = k_m * dT_dz

    f_rho_z = jnp.zeros_like(q)
    f_m1_z = jnp.zeros_like(q)
    f_m2_z = jnp.zeros_like(q)
    f_m3_z = jnp.zeros_like(q)
    f_E_z = q

    F = jnp.stack((f_rho_z, f_m1_z, f_m2_z, f_m3_z, f_E_z), axis = 0)
    return d_x * d_y * (F[:, :, :, 1:] - F[:, :, :, :-1])

def div_naive_heat_flux_3d(u, T):
    return div_x_naive_heat_flux_3d(u, T) + div_y_naive_heat_flux_3d(u, T) + div_z_naive_heat_flux_3d(u, T)



''' 2D versions of naive heat flux divergence '''

def div_x_naive_heat_flux_2d(u, T):
    '''
    Assume u is padded appropriately (5, n_x + 2, n_y + 2)
    '''

    n_x, n_y = GRID_RESOLUTION
    d_x, d_y = GRID_SPACING

    k = thermal_conductivity(u, T)
    k_m = 0.5 * (k[1:, 1:-1] + k[:-1, 1:-1])

    dT_dx = (T[1:, 1:-1] - T[:-1, 1:-1]) / d_x
    q = k_m * dT_dx

    f_rho_x = jnp.zeros((n_x + 1, n_y))
    f_m1_x = jnp.zeros((n_x + 1, n_y))
    f_m2_x = jnp.zeros((n_x + 1, n_y))
    f_E_x = q

    F = jnp.stack((f_rho_x, f_m1_x, f_m2_x, f_E_x), axis = 0)
    return d_y * (F[:, 1:, :] - F[:, :-1, :])

def div_y_naive_heat_flux_2d(u, T):
    '''
    Assume u is padded appropriately (5, n_x + 2, n_y + 2)
    '''

    n_x, n_y = GRID_RESOLUTION
    d_x, d_y = GRID_SPACING

    k = thermal_conductivity(u, T)
    k_m = 0.5 * (k[1:-1, 1:] + k[1:-1, :-1])

    dT_dy = (T[1:-1, 1:] - T[1:-1, :-1]) / d_y
    q = k_m * dT_dy

    f_rho_y = jnp.zeros((n_x, n_y + 1))
    f_m1_y = jnp.zeros((n_x, n_y + 1))
    f_m2_y = jnp.zeros((n_x, n_y + 1))
    f_E_y = q

    F = jnp.stack((f_rho_y, f_m1_y, f_m2_y, f_E_y), axis = 0)
    return d_x * (F[:, :, 1:] - F[:, :, :-1])

def div_naive_heat_flux_2d(u, T):
    return div_x_naive_heat_flux_2d(u, T) + div_y_naive_heat_flux_2d(u, T)


''' 1D version of naive heat flux divergence'''
def div_naive_heat_flux_1d(u, T):
    '''
    Assume u is padded appropriately (5, n_x + 2)
    '''

    n_x = GRID_RESOLUTION[0]
    d_x = GRID_SPACING[0]

    k = thermal_conductivity(u, T)
    k_m = 0.5 * (k[1:] + k[:-1])

    dT_dx = (T[1:] - T[:-1]) / d_x
    q = k_m * dT_dx

    f_rho_x = jnp.zeros((n_x + 1))
    f_m_x = jnp.zeros((n_x + 1))
    f_E_x = q

    F = jnp.stack((f_rho_x, f_m_x, f_E_x), axis = 0)
    return F[:, 1:] - F[:, :-1]