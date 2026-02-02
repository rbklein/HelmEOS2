"""
    All functions for the naive flux are gathered here.
    This flux is a simple implementation that does not consider any physical properties or
    numerical stability. It is primarily used for testing and debugging purposes.
"""

from prep_jax import *
from config.conf_numerical import *

from modules.geometry.grid      import GRID_RESOLUTION, GRID_SPACING
from modules.thermodynamics.EOS import pressure
from jax.numpy                  import stack

''' 3D versions of the naive fluxes for testing purposes '''

def div_x_naive_3d(u, T):
    """
    Assume u is padded appropriately (5, n_x + 2, n_y + 2, n_z + 2).
    """

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    u_m_x = 0.5 * (u[:,1:, :, :]  + u[:,:-1, :, :])  # arithmetic mean in x-direction  
    T = 0.5 * (T[1:, :, :] + T[:-1,:,:])
    p = pressure(u_m_x[0], T)
    f_rho_x = u_m_x[1]
    f_m1_x = u_m_x[1]**2 / u_m_x[0] + p
    f_m2_x = u_m_x[1] * u_m_x[2] / u_m_x[0]
    f_m3_x = u_m_x[1] * u_m_x[3] / u_m_x[0]
    f_E_x = u_m_x[1] * (u_m_x[4] + p) / u_m_x[0]

    F = stack((f_rho_x, f_m1_x, f_m2_x, f_m3_x, f_E_x), axis=0)
    return d_y * d_z * (F[:, 1:, 1:(n_y+1), 1:(n_z+1)] - F[:, :-1, 1:(n_y+1), 1:(n_z+1)])  # Return the difference in fluxes in x-direction

def div_y_naive_3d(u, T):
    """
    Assume u is padded appropriately (5, n_x + 2, n_y + 2, n_z + 2).
    """

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    u_m_y = 0.5 * (u[:, :, 1:, :] + u[:, :, :-1, :])  # arithmetic mean in y-direction
    T = 0.5 * (T[:, 1:, :] + T[:, :-1, :])
    p = pressure(u_m_y[0], T)
    f_rho_y = u_m_y[2]
    f_m1_y = u_m_y[1] * u_m_y[2] / u_m_y[0]
    f_m2_y = u_m_y[2]**2 / u_m_y[0] + p
    f_m3_y = u_m_y[2] * u_m_y[3] / u_m_y[0]
    f_E_y = u_m_y[2] * (u_m_y[4] + p) / u_m_y[0]

    F = stack((f_rho_y, f_m1_y, f_m2_y, f_m3_y, f_E_y), axis=0)
    return d_x * d_z * (F[:, 1:(n_x+1), 1:, 1:(n_z+1)] - F[:, 1:(n_x+1), :-1, 1:(n_z+1)])  # Return the difference in fluxes in y-direction

def div_z_naive_3d(u, T):
    """
    Assume u is padded appropriately (5, n_x + 2, n_y + 2, n_z + 2).
    """

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    u_m_z = 0.5 * (u[:, :, :, 1:] + u[:, :, :, :-1])  # arithmetic mean in z-direction
    T = 0.5 * (T[:, :, 1:] + T[:, :, :-1])
    p = pressure(u_m_z[0], T)
    f_rho_z = u_m_z[3]
    f_m1_z = u_m_z[1] * u_m_z[3] / u_m_z[0]
    f_m2_z = u_m_z[2] * u_m_z[3] / u_m_z[0]
    f_m3_z = u_m_z[3]**2 / u_m_z[0] + p
    f_E_z = u_m_z[3] * (u_m_z[4] + p) / u_m_z[0]

    F = stack((f_rho_z, f_m1_z, f_m2_z, f_m3_z, f_E_z), axis=0)
    return d_x * d_y * (F[:, 1:(n_x+1), 1:(n_y+1), 1:] - F[:, 1:(n_x+1), 1:(n_y+1), :-1])  # Return the difference in fluxes in z-direction

def div_naive_3d(u, T):
    return div_x_naive_3d(u, T) + div_y_naive_3d(u, T) + div_z_naive_3d(u, T)


''' 2D versions of the naive fluxes for testing purposes '''

def div_x_naive_2d(u, T):
    """
    Assume u is padded appropriately (5, n_x + 2, n_y + 2).
    """

    n_x, n_y = GRID_RESOLUTION
    d_x, d_y = GRID_SPACING

    u_m_x = 0.5 * (u[:,1:, :] + u[:,:-1, :])  # arithmetic mean in x-direction  
    T = 0.5 * (T[1:, :] + T[:-1, :])
    p = pressure(u_m_x[0], T)
    f_rho_x = u_m_x[1]
    f_m1_x = u_m_x[1]**2 / u_m_x[0] + p
    f_m2_x = u_m_x[1] * u_m_x[2] / u_m_x[0]
    f_E_x = u_m_x[1] * (u_m_x[3] + p) / u_m_x[0]

    F = stack((f_rho_x, f_m1_x, f_m2_x, f_E_x), axis=0)
    return d_y * (F[:, 1:, 1:(n_y+1)] - F[:, :-1, 1:(n_y+1)])  # Return the difference in fluxes in x-direction

def div_y_naive_2d(u, T):
    """
    Assume u is padded appropriately (5, n_x + 2, n_y + 2).
    """

    n_x, n_y = GRID_RESOLUTION
    d_x, d_y = GRID_SPACING

    u_m_y = 0.5 * (u[:, :, 1:] + u[:, :, :-1])  # arithmetic mean in y-direction
    T = 0.5 * (T[:, 1:] + T[:, :-1])
    p = pressure(u_m_y[0], T)
    f_rho_y = u_m_y[2]
    f_m1_y = u_m_y[1] * u_m_y[2] / u_m_y[0]
    f_m2_y = u_m_y[2]**2 / u_m_y[0] + p
    f_E_y = u_m_y[2] * (u_m_y[3] + p) / u_m_y[0]

    F = stack((f_rho_y, f_m1_y, f_m2_y, f_E_y), axis=0)
    return d_x * (F[:, 1:(n_x+1), 1:] - F[:, 1:(n_x+1), :-1])  # Return the difference in fluxes in y-direction

    
def div_naive_2d(u, T):
    return div_x_naive_2d(u, T) + div_y_naive_2d(u, T)







