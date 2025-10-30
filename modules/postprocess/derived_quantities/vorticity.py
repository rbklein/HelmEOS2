"""
    This file contains functions to compute the vorticity or curl of the velocity field
"""

from prep_jax import *
from modules.geometry.grid import *
from modules.simulation.boundary import apply_boundary_conditions

def vorticity_2d(u, T):
    """
        Calculate the vorticity of the 2D velocity field u
    """

    dx, dy = GRID_SPACING

    u = apply_boundary_conditions(u, T)
    vel = u[1:3, :, :] / u[0, : ,:]

    vel_x_y = (vel[0, 1:-1, 2:] - vel[0, 1:-1, :-2]) / (2 * dy)
    vel_y_x = (vel[1, 2:, 1:-1] - vel[1, :-2, 1:-1]) / (2 * dx)

    return vel_y_x - vel_x_y

def vorticity_3d(u, T):
    """
        Calculate the vorticity of the 3D velocity field u
    """

    dx, dy, dz = GRID_SPACING

    u = apply_boundary_conditions(u, T)
    vel = u[1:4, :, :] / u[0, : ,:]

    vel_x_y = (vel[0, 1:-1, 2:, 1:-1] - vel[0, 1:-1, :-2, 1:-1]) / (2 * dy)
    vel_x_z = (vel[0, 1:-1, 1:-1, 2:] - vel[0, 1:-1, 1:-1, :-2]) / (2 * dz)

    vel_y_x = (vel[1, 2:, 1:-1, 1:-1] - vel[1, :-2, 1:-1, 1:-1]) / (2 * dx)
    vel_y_z = (vel[1, 1:-1, 1:-1, 2:] - vel[1, 1:-1, 1:-1, :-2]) / (2 * dz)

    vel_z_x = (vel[2, 2:, 1:-1, 1:-1] - vel[2, :-2, 1:-1, 1:-1]) / (2 * dx)
    vel_z_y = (vel[2, 1:-1, 2:, 1:-1] - vel[2, 1:-1, :-2, 1:-1]) / (2 * dy)

    curl_x = vel_z_y - vel_y_z
    curl_y = vel_x_z - vel_z_x
    curl_z = vel_y_x - vel_x_y

    return jnp.stack((curl_x, curl_y, curl_z), axis = 0)

if N_DIMENSIONS == 2:
    vorticity = vorticity_2d
elif N_DIMENSIONS == 3:
    vorticity = vorticity_3d
