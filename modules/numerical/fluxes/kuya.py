from prep_jax import *
from config.conf_numerical      import *
from config.conf_thermodynamics import *

from modules.geometry.grid          import GRID_RESOLUTION, GRID_SPACING
from modules.thermodynamics.EOS     import pressure, internal_energy
from jax.numpy                      import stack, sum

def div_kuya_1d(u, T):
    rho = u[0]
    vel = u[1] / rho 
    p   = pressure(rho, T)
    e   = internal_energy(rho, T)

    rho_m = 0.5 * (rho[1:] + rho[:-1])
    vel_m = 0.5 * (vel[1:] + vel[:-1])

    f_rho_x = rho_m * vel_m

    p_m = 0.5 * (p[1:] + p[:-1])

    f_m1_x = f_rho_x * vel_m + p_m

    e_m = 0.5 * (e[1:] + e[:-1])
    k_m = vel[1:] * vel[:-1] / 2
    pu_m = 0.5 * (p[1:] * vel[:-1] + p[:-1] * vel[1:])
    
    f_e = f_rho_x * e_m 
    f_k = f_rho_x * k_m

    f_E_x = f_e + f_k + pu_m

    F = stack((f_rho_x, f_m1_x, f_E_x), axis=0)
    return (F[:, 1:] - F[:, :-1])




def div_x_kuya_2d(u, T):
    """
    Kuya's KEEP-Q flux: 
    
    Comprehensive analysis of entropy conservation property of 
    non-dissipative schemes for compressible flows: KEEP scheme redefined
    """

    n_x, n_y = GRID_RESOLUTION
    d_x, d_y = GRID_SPACING

    rho = u[0]
    vel = u[1:3] / rho 
    p   = pressure(rho, T)
    e   = internal_energy(rho, T)

    rho_m = 0.5 * (rho[1:, :] + rho[:-1, :])
    vel_m = 0.5 * (vel[:, 1:, :] + vel[:, :-1, :])

    f_rho_x = rho_m * vel_m[0]

    p_m = 0.5 * (p[1:, :] + p[:-1, :])

    f_m1_x = f_rho_x * vel_m[0] + p_m
    f_m2_x = f_rho_x * vel_m[1]

    e_m = 0.5 * (e[1:, :] + e[:-1, :])
    k_m = sum(vel[:, 1:, :] * vel[:, :-1, :], axis = 0) / 2
    pu_m = 0.5 * (p[1:, :] * vel[0, :-1, :] + p[:-1, :] * vel[0, 1:, :])
    
    f_e = f_rho_x * e_m 
    f_k = f_rho_x * k_m

    f_E_x = f_e + f_k + pu_m

    F = stack((f_rho_x, f_m1_x, f_m2_x, f_E_x), axis=0)
    return d_y * (F[:, 1:, 1:(n_y+1)] - F[:, :-1, 1:(n_y+1)])  # Return the difference in fluxes in x-direction



def div_y_kuya_2d(u, T):
    """
    Kuya's KEEP-Q flux: 
    
    Comprehensive analysis of entropy conservation property of 
    non-dissipative schemes for compressible flows: KEEP scheme redefined
    """

    n_x, n_y = GRID_RESOLUTION
    d_x, d_y = GRID_SPACING

    rho = u[0]
    vel = u[1:3] / rho 
    p   = pressure(rho, T)
    e   = internal_energy(rho, T)

    rho_m = 0.5 * (rho[:, 1:] + rho[:, :-1])
    vel_m = 0.5 * (vel[:, :, 1:] + vel[:, :, :-1])

    f_rho_y = rho_m * vel_m[1]

    p_m = 0.5 * (p[:, 1:] + p[:, :-1])
    f_m1_y = f_rho_y * vel_m[0] 
    f_m2_y = f_rho_y * vel_m[1] + p_m

    e_m = 0.5 * (e[:, 1:] + e[:, :-1])
    k_m = sum(vel[:, :, 1:] * vel[:, :, :-1], axis = 0) / 2
    pu_m = 0.5 * (p[:, 1:] * vel[1, :, :-1] + p[:, :-1] * vel[1, :, 1:])
    
    f_e = f_rho_y * e_m 
    f_k = f_rho_y * k_m

    f_E_y = f_e + f_k + pu_m

    F = stack((f_rho_y, f_m1_y, f_m2_y, f_E_y), axis=0)
    return d_x * (F[:, 1:(n_x+1), 1:] - F[:, 1:(n_x+1), :-1])  # Return the difference in fluxes in y-direction

def div_kuya_2d(u, T):
    return div_x_kuya_2d(u, T) + div_y_kuya_2d(u, T)



def div_x_kuya_3d(u, T):
    """
    Kuya's KEEP-Q flux: 
    
    Comprehensive analysis of entropy conservation property of 
    non-dissipative schemes for compressible flows: KEEP scheme redefined
    """

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    rho = u[0]
    vel = u[1:4] / rho 
    p   = pressure(rho, T)
    e   = internal_energy(rho, T)

    rho_m = 0.5 * (rho[1:, :, :] + rho[:-1, :, :])
    vel_m = 0.5 * (vel[:, 1:, :, :] + vel[:, :-1, :, :])

    f_rho_x = rho_m * vel_m[0]

    p_m = 0.5 * (p[1:, :, :] + p[:-1, :, :])

    f_m1_x = f_rho_x * vel_m[0] + p_m
    f_m2_x = f_rho_x * vel_m[1]
    f_m3_x = f_rho_x * vel_m[2]

    e_m = 0.5 * (e[1:, :, :] + e[:-1, :, :])
    k_m = sum(vel[:, 1:, :, :] * vel[:, :-1, :, :], axis = 0) / 2
    pu_m = 0.5 * (p[1:, :, :] * vel[0, :-1, :, :] + p[:-1, :, :] * vel[0, 1:, :, :])
    
    f_e = f_rho_x * e_m 
    f_k = f_rho_x * k_m

    f_E_x = f_e + f_k + pu_m

    F = stack((f_rho_x, f_m1_x, f_m2_x, f_m3_x, f_E_x), axis=0)
    return d_y * d_z * (F[:, 1:, 1:(n_y+1), 1:(n_z+1)] - F[:, :-1, 1:(n_y+1), 1:(n_z+1)])  # Return the difference in fluxes in x-direction


def div_y_kuya_3d(u, T):
    """
    Kuya's KEEP-Q flux: 
    
    Comprehensive analysis of entropy conservation property of 
    non-dissipative schemes for compressible flows: KEEP scheme redefined
    """

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    rho = u[0]
    vel = u[1:4] / rho 
    p   = pressure(rho, T)
    e   = internal_energy(rho, T)

    rho_m = 0.5 * (rho[:, 1:, :] + rho[:, :-1, :])
    vel_m = 0.5 * (vel[:, :, 1:, :] + vel[:, :, :-1, :])

    f_rho_y = rho_m * vel_m[1]

    p_m = 0.5 * (p[:, 1:, :] + p[:, :-1, :])
    f_m1_y = f_rho_y * vel_m[0] 
    f_m2_y = f_rho_y * vel_m[1] + p_m
    f_m3_y = f_rho_y * vel_m[2] 

    e_m = 0.5 * (e[:, 1:, :] + e[:, :-1, :])
    k_m = sum(vel[:, :, 1:, :] * vel[:, :, :-1, :], axis = 0) / 2
    pu_m = 0.5 * (p[:, 1:, :] * vel[1, :, :-1, :] + p[:, :-1, :] * vel[1, :, 1:, :])
    
    f_e = f_rho_y * e_m 
    f_k = f_rho_y * k_m

    f_E_y = f_e + f_k + pu_m

    F = stack((f_rho_y, f_m1_y, f_m2_y, f_m3_y, f_E_y), axis=0)
    return d_x * d_z * (F[:, 1:(n_x+1), 1:, 1:(n_z+1)] - F[:, 1:(n_x+1), :-1, 1:(n_z+1)])  # Return the difference in fluxes in y-direction


def div_z_kuya_3d(u, T):
    """
    Kuya's KEEP-Q flux: 
    
    Comprehensive analysis of entropy conservation property of 
    non-dissipative schemes for compressible flows: KEEP scheme redefined
    """

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    rho = u[0]
    vel = u[1:4] / rho 
    p   = pressure(rho, T)
    e   = internal_energy(rho, T)

    rho_m = 0.5 * (rho[:, :, 1:] + rho[:, :, :-1])
    vel_m = 0.5 * (vel[:, :, :, 1:] + vel[:, :, :, :-1])

    f_rho_z = rho_m * vel_m[2]

    p_m = 0.5 * (p[:, :, 1:] + p[:, :, :-1])
    f_m1_z = f_rho_z * vel_m[0] 
    f_m2_z = f_rho_z * vel_m[1] 
    f_m3_z = f_rho_z * vel_m[2] + p_m

    e_m = 0.5 * (e[:, :, 1:] + e[:, :, :-1])
    k_m = sum(vel[:, :, :, 1:] * vel[:, :, :, :-1], axis = 0) / 2
    pu_m = 0.5 * (p[:, :, 1:] * vel[2, :, :, :-1] + p[:, :, :-1] * vel[2, :, :, 1:])
    
    f_e = f_rho_z * e_m 
    f_k = f_rho_z * k_m

    f_E_z = f_e + f_k + pu_m

    F = stack((f_rho_z, f_m1_z, f_m2_z, f_m3_z, f_E_z), axis=0)
    return d_x * d_y * (F[:, 1:(n_x+1), 1:(n_y+1), 1:] - F[:, 1:(n_x+1), 1:(n_y+1), :-1])  # Return the difference in fluxes in y-direction


def div_kuya_3d(u, T):
    return div_x_kuya_3d(u, T) + div_y_kuya_3d(u, T) + div_z_kuya_3d(u, T)