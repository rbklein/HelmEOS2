"""
    Implementation of a naive viscous flux
"""

from prep_jax import *
from config.conf_numerical import *
from modules.geometry.grid import *
from modules.thermodynamics.EOS import *
from modules.thermodynamics.constitutive import *

''' 3D Versions of naive stress tensor divergence '''

def div_x_naive_stress_3d(u, T):
    '''
        Assume u is padded appropriately (5, n_x + 2, n_y + 2, n_z + 2)
    '''

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    mu  = dynamic_viscosity(T)
    mu_m = 0.5 * (mu[1:, 1:-1, 1:-1] + mu[:-1, 1:-1, 1:-1])  #(n_x + 1, n_y, n_z) mean dynamic viscosity

    zeta = bulk_viscosity(T)
    zeta_m = 0.5 * (zeta[1:, 1:-1, 1:-1] + zeta[:-1, 1:-1, 1:-1])  #(n_x + 1, n_y, n_z) mean bulk viscosity

    ''' Du + Du^T '''
    vel = u[1:4] / u[0]
    d_vel_dx = (vel[:, 1:, 1:-1, 1:-1] - vel[:, :-1, 1:-1, 1:-1]) / d_x     #(3, n_x + 1, n_y, n_z)
    du_dx = d_vel_dx[0]                                                     #(n_x + 1, n_y, n_z)
    
    #compute difference at i,j,k by subtracting values at i,j+1,k and i,j-1,k
    du_dy = (vel[0, :, 2:, :] - vel[0, :, :-2, :]) / (2 * d_y)      #(n_x + 2, n_y, n_z + 2)
    du_dy_m = 0.5 * (du_dy[1:, : , 1:-1] + du_dy[:-1, :, 1:-1])     #(n_x + 1, n_y, n_z)

    du_dz = (vel[0, :, :, 2:] - vel[0, :, :, :-2]) / (2 * d_z)      #(n_x + 2, n_y + 2, n_z)
    du_dz_m = 0.5 * (du_dz[1:, 1:-1, :] + du_dz[:-1, 1:-1 ,:])      #(n_x + 1, n_y, n_z)

    ''' div(u) I '''
    dv_dy = (vel[1, :, 2:, :] - vel[1, :, :-2, :]) / (2 * d_y)      #(n_x + 2, n_y, n_z + 2)
    dv_dy_m = 0.5 * (dv_dy[1:, : , 1:-1] + dv_dy[:-1, :, 1:-1])     #(n_x + 1, n_y, n_z)
    dw_dz = (vel[2, :, :, 2:] - vel[2, :, :, :-2]) / (2 * d_z)      #(n_x + 2, n_y + 2, n_z)
    dw_dz_m = 0.5 * (dw_dz[1:, 1:-1, :] + dw_dz[:-1, 1:-1 ,:])      #(n_x + 1, n_y, n_z)

    div_vel = du_dx + dv_dy_m + dw_dz_m #(n_x + 1, n_y, n_z)

    vel_m = 0.5 * (vel[:, 1:, 1:-1, 1:-1] + vel[:, :-1, 1:-1, 1:-1])

    f_rho_x = jnp.zeros((n_x + 1, n_y, n_z))
    f_m1_x = mu_m * (du_dx + d_vel_dx[0]) + (zeta_m - 2/3 * mu_m) * div_vel     #compute second viscosity from bulk and dynamic viscosity
    f_m2_x = mu_m * (du_dy_m + d_vel_dx[1])
    f_m3_x = mu_m * (du_dz_m + d_vel_dx[2])
    f_E_x = mu_m * (vel_m[0] * f_m1_x + vel_m[1] * f_m2_x + vel_m[2] * f_m3_x) 

    F = jnp.stack((f_rho_x, f_m1_x, f_m2_x, f_m3_x, f_E_x), axis=0)
    return d_y * d_z * (F[:, 1:, :, :] - F[:, :-1, :, :])  # Return the difference in fluxes in x-direction

def div_y_naive_stress_3d(u, T):
    '''
        Assume u is padded appropriately (5, n_x + 2, n_y + 2, n_z + 2)
    '''

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    mu  = dynamic_viscosity(T)
    mu_m = 0.5 * (mu[1:-1, 1:, 1:-1] + mu[1:-1, :-1, 1:-1])  #(n_x, n_y + 1, n_z) mean dynamic viscosity

    zeta = bulk_viscosity(T)
    zeta_m = 0.5 * (zeta[1:-1, 1:, 1:-1] + zeta[1:-1, :-1, 1:-1])  #(n_x, n_y + 1, n_z) mean bulk viscosity

    ''' Du + Du^T '''
    vel = u[1:4] / u[0]
    d_vel_dy = (vel[:, 1:-1, 1:, 1:-1] - vel[:, 1:-1, :-1, 1:-1]) / d_y     #(3, n_x, n_y + 1, n_z)
    dv_dy = d_vel_dy[1]                                                     #(n_x, n_y+1, n_z)

    #differentiate in x then average to cell face in y
    dv_dx = (vel[1, 2:, :, :] - vel[1, :-2, :, :]) / (2 * d_x)     #(n_x, n_y + 2, n_z + 2)
    dv_dx_m = 0.5 * (dv_dx[:, 1:, 1:-1] + dv_dx[:, :-1, 1:-1])     #(n_x, n_y + 1, n_z)

    dv_dz = (vel[1, :, :, 2:] - vel[1, :, :, :-2]) / (2 * d_z)      #(n_x + 2, n_y + 2, n_z)
    dv_dz_m = 0.5 * (dv_dz[1:-1, 1:, :] + dv_dz[1:-1, :-1 ,:])      #(n_x, n_y + 1, n_z)

    ''' div(u) I '''
    du_dx = (vel[0, 2:, :, :] - vel[0, :-2, :, :]) / (2 * d_x)     #(n_x, n_y + 2, n_z + 2)
    du_dx_m = 0.5 * (du_dx[:, 1:, 1:-1] + du_dx[:, :-1, 1:-1])     #(n_x, n_y + 1, n_z)

    dw_dz = (vel[2, :, :, 2:] - vel[2, :, :, :-2]) / (2 * d_z)      #(n_x + 2, n_y + 2, n_z)
    dw_dz_m = 0.5 * (dw_dz[1:-1, 1:, :] + dw_dz[1:-1, :-1 ,:])      #(n_x, n_y + 1, n_z)

    div_vel = du_dx_m + dv_dy + dw_dz_m

    vel_m = 0.5 * (vel[:, 1:-1, 1:, 1:-1] + vel[:, 1:-1, :-1, 1:-1])

    f_rho_y = jnp.zeros((n_x, n_y + 1, n_z))
    f_m1_y = mu_m * (dv_dx_m + d_vel_dy[0]) 
    f_m2_y = mu_m * (dv_dy + d_vel_dy[1]) + (zeta_m - 2/3 * mu_m) * div_vel     #compute second viscosity from bulk and dynamic viscosity
    f_m3_y = mu_m * (dv_dz_m + d_vel_dy[2])
    f_E_y = mu_m * (vel_m[0] * f_m1_y + vel_m[1] * f_m2_y + vel_m[2] * f_m3_y) 

    F = jnp.stack((f_rho_y, f_m1_y, f_m2_y, f_m3_y, f_E_y), axis=0)
    return d_x * d_z * (F[:, :, 1:, :] - F[:, :, :-1, :])  # Return the difference in fluxes in y-direction

def div_z_naive_stress_3d(u, T):
    '''
        Assume u is padded appropriately (5, n_x + 2, n_y + 2, n_z + 2)
    '''

    n_x, n_y, n_z = GRID_RESOLUTION
    d_x, d_y, d_z = GRID_SPACING

    mu  = dynamic_viscosity(T)
    mu_m = 0.5 * (mu[1:-1, 1:-1, 1:] + mu[1:-1, 1:-1, :-1])  #(n_x, n_y, n_z + 1) mean dynamic viscosity

    zeta = bulk_viscosity(T)
    zeta_m = 0.5 * (zeta[1:-1, 1:-1, 1:] + zeta[1:-1, 1:-1, :-1])  #(n_x, n_y, n_z + 1) mean bulk viscosity

    ''' Du + Du^T '''
    vel = u[1:4] / u[0]
    d_vel_dz = (vel[:, 1:-1, 1:-1, 1:] - vel[:, 1:-1, 1:-1, :-1]) / d_z #(3, n_x, n_y, n_z + 1)
    dw_dz = d_vel_dz[2]

    dw_dx = (vel[2, 2:, :, :] - vel[2, :-2, :, :]) / (2 * d_x) #(n_x, n_y + 2, n_z + 2)
    dw_dx_m = 0.5 * (dw_dx[:, 1:-1, 1:] + dw_dx[:, 1:-1, :-1]) #(n_x, n_y, n_z + 1)

    dw_dy = (vel[2, :, 2:, :] - vel[2, :, :-2, :]) / (2 * d_y) #(n_x + 2, n_y, n_z + 2)
    dw_dy_m = 0.5 * (dw_dy[1:-1, :, 1:] + dw_dy[1:-1, :, :-1]) #(n_x, n_y, n_z + 1)

    ''' div(u) I '''
    du_dx = (vel[0, 2:, :, :] - vel[0, :-2, :, :]) / (2 * d_x) #(n_x, n_y + 2, n_z + 2)
    du_dx_m = 0.5 * (du_dx[:, 1:-1, 1:] + du_dx[:, 1:-1, :-1]) #(n_x, n_y, n_z + 1)

    dv_dy = (vel[1, :, 2:, :] - vel[1, :, :-2, :]) / (2 * d_y) #(n_x + 2, n_y, n_z + 2)
    dv_dy_m = 0.5 * (dv_dy[1:-1, :, 1:] + dv_dy[1:-1, :, :-1]) #(n_x, n_y, n_z + 1)

    div_vel = du_dx_m + dv_dy_m + dw_dz

    vel_m = 0.5 * (vel[:, 1:-1, 1:-1, 1:] + vel[:, 1:-1, 1:-1, :-1])

    f_rho_z = jnp.zeros((n_x, n_y, n_z + 1))
    f_m1_z = mu_m * (dw_dx_m + d_vel_dz[0]) 
    f_m2_z = mu_m * (dw_dy_m + d_vel_dz[1]) 
    f_m3_z = mu_m * (dw_dz + d_vel_dz[2]) + (zeta_m - 2/3 * mu_m) * div_vel     #compute second viscosity from bulk and dynamic viscosity
    f_E_y = mu_m * (vel_m[0] * f_m1_z + vel_m[1] * f_m2_z + vel_m[2] * f_m3_z) 

    F = jnp.stack((f_rho_z, f_m1_z, f_m2_z, f_m3_z, f_E_y), axis=0)
    return d_x * d_y * (F[:, :, :, 1:] - F[:, :, :, :-1])  # Return the difference in fluxes in y-direction

def div_naive_stress_3d(u, T):
    return div_x_naive_stress_3d(u, T) + div_y_naive_stress_3d(u, T) + div_z_naive_stress_3d(u, T)



''' 2D versions of naive stress tensor divergence '''

def div_x_naive_stress_2d(u, T):
    '''
        Assume u is padded appropriately (5, n_x + 2, n_y + 2)
    '''

    n_x, n_y = GRID_RESOLUTION
    d_x, d_y = GRID_SPACING

    mu  = dynamic_viscosity(T)
    mu_m = 0.5 * (mu[1:, 1:-1] + mu[:-1, 1:-1])  #(n_x + 1, n_y) mean dynamic viscosity

    zeta = bulk_viscosity(T)
    zeta_m = 0.5 * (zeta[1:, 1:-1] + zeta[:-1, 1:-1])  #(n_x + 1, n_y) mean bulk viscosity

    ''' Du + Du^T '''
    vel = u[1:3] / u[0]
    d_vel_dx = (vel[:, 1:, 1:-1] - vel[:, :-1, 1:-1]) / d_x     #(2, n_x + 1, n_y)
    du_dx = d_vel_dx[0]                                         #(n_x + 1, n_y)
    
    #compute difference at i,j by subtracting values at i,j+1 and i,j-1
    du_dy = (vel[0, :, 2:] - vel[0, :, :-2]) / (2 * d_y)        #(n_x + 2, n_y)
    du_dy_m = 0.5 * (du_dy[1:, :] + du_dy[:-1, :])              #(n_x + 1, n_y)


    ''' div(u) I '''
    dv_dy = (vel[1, :, 2:] - vel[1, :, :-2]) / (2 * d_y)        #(n_x + 2, n_y)
    dv_dy_m = 0.5 * (dv_dy[1:, :] + dv_dy[:-1, :])              #(n_x + 1, n_y)

    div_vel = du_dx + dv_dy_m #(n_x + 1, n_y)

    vel_m = 0.5 * (vel[:, 1:, 1:-1] + vel[:, :-1, 1:-1])

    f_rho_x = jnp.zeros((n_x + 1, n_y))
    f_m1_x = mu_m * (du_dx + d_vel_dx[0]) + (zeta_m - mu_m) * div_vel     #compute second viscosity from bulk and dynamic viscosity
    f_m2_x = mu_m * (du_dy_m + d_vel_dx[1])
    f_E_x = mu_m * (vel_m[0] * f_m1_x + vel_m[1] * f_m2_x) 

    F = jnp.stack((f_rho_x, f_m1_x, f_m2_x, f_E_x), axis=0)
    return d_y * (F[:, 1:, :] - F[:, :-1, :])  # Return the difference in fluxes in x-direction

def div_y_naive_stress_2d(u, T):
    '''
        Assume u is padded appropriately (5, n_x + 2, n_y + 2)
    '''

    n_x, n_y = GRID_RESOLUTION
    d_x, d_y = GRID_SPACING

    mu  = dynamic_viscosity(T)
    mu_m = 0.5 * (mu[1:-1, 1:] + mu[1:-1, :-1])  #(n_x, n_y + 1) mean dynamic viscosity

    zeta = bulk_viscosity(T)
    zeta_m = 0.5 * (zeta[1:-1, 1:] + zeta[1:-1, :-1])  #(n_x, n_y + 1) mean bulk viscosity

    ''' Du + Du^T '''
    vel = u[1:3] / u[0]
    d_vel_dy = (vel[:, 1:-1, 1:] - vel[:, 1:-1, :-1]) / d_y     #(3, n_x, n_y + 1)
    dv_dy = d_vel_dy[1]                                         #(n_x, n_y+1)

    #differentiate in x then average to cell face in y
    dv_dx = (vel[1, 2:, :] - vel[1, :-2, :]) / (2 * d_x)    #(n_x, n_y + 2)
    dv_dx_m = 0.5 * (dv_dx[:, 1:] + dv_dx[:, :-1])          #(n_x, n_y + 1)

    ''' div(u) I '''
    du_dx = (vel[0, 2:, :] - vel[0, :-2, :]) / (2 * d_x)    #(n_x, n_y + 2)
    du_dx_m = 0.5 * (du_dx[:, 1:] + du_dx[:, :-1])          #(n_x, n_y + 1)

    div_vel = du_dx_m + dv_dy 

    vel_m = 0.5 * (vel[:, 1:-1, 1:] + vel[:, 1:-1, :-1])

    f_rho_y = jnp.zeros((n_x, n_y + 1))
    f_m1_y = mu_m * (dv_dx_m + d_vel_dy[0]) 
    f_m2_y = mu_m * (dv_dy + d_vel_dy[1]) + (zeta_m - mu_m) * div_vel     #compute second viscosity from bulk and dynamic viscosity
    f_E_y = mu_m * (vel_m[0] * f_m1_y + vel_m[1] * f_m2_y) 

    F = jnp.stack((f_rho_y, f_m1_y, f_m2_y, f_E_y), axis=0)
    return d_x * (F[:, :, 1:] - F[:, :, :-1])  # Return the difference in fluxes in y-direction

def div_naive_stress_2d(u, T):
    return div_x_naive_stress_2d(u, T) + div_y_naive_stress_2d(u, T)



''' 1D versions of naive stress tensor divergence '''

def div_naive_stress_1d(u, T):
    '''
        Assume u is padded appropriately (5, n_x + 2)
    '''

    n_x = GRID_RESOLUTION[0]
    d_x = GRID_SPACING[0]

    mu  = dynamic_viscosity(T)
    mu_m = 0.5 * (mu[1:] + mu[:-1])  #(n_x + 1) mean dynamic viscosity

    ''' Du + Du^T '''
    vel = u[1] / u[0]
    d_vel_dx = (vel[1:] - vel[:-1]) / d_x     #(2, n_x + 1, n_y)
    vel_m = 0.5 * (vel[1:] + vel[:-1])

    f_rho_x = jnp.zeros((n_x + 1))
    f_m1_x = mu_m * d_vel_dx
    f_E_x = mu_m * vel_m * f_m1_x 

    F = jnp.stack((f_rho_x, f_m1_x, f_E_x), axis=0)
    return F[:, 1:] - F[:, :-1]  # Return the difference in fluxes in x-direction