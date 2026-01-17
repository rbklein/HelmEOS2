"""
    This file contains functions to compute the vorticity or curl of the velocity field
"""

from prep_jax               import *
from config.conf_geometry   import *

from modules.geometry.grid          import GRID_SPACING, CELL_VOLUME
from modules.simulation.boundary    import apply_boundary_conditions
from modules.thermodynamics.EOS     import entropy, kinetic_energy, pressure
from modules.thermodynamics.constitutive import dynamic_viscosity
from jax.numpy                      import stack, sum

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

    return stack((curl_x, curl_y, curl_z), axis = 0)

def divergence_3d(u, T):
    """
        Calculate the divergence of the 3D velocity field u
    """

    dx, dy, dz = GRID_SPACING

    u = apply_boundary_conditions(u, T)
    vel = u[1:4, :, :] / u[0, : ,:]

    vel_x_x = (vel[0, 2:, 1:-1, 1:-1] - vel[0, :-2, 1:-1, 1:-1]) / (2 * dx)
    vel_y_y = (vel[1, 1:-1, 2:, 1:-1] - vel[1, 1:-1, :-2, 1:-1]) / (2 * dy)
    vel_z_z = (vel[2, 1:-1, 1:-1, 2:] - vel[2, 1:-1, 1:-1, :-2]) / (2 * dz)

    return vel_x_x + vel_y_y + vel_z_z

def pressure_work(u, T):
    div = divergence_3d(u, T)
    p   = pressure(u[0], T)
    return sum(p * div) * CELL_VOLUME

def total_dilation(u, T):
    div = divergence_3d(u, T)
    mu  = dynamic_viscosity(u, T)
    p   = 4/3 * mu * div**2
    return sum(p) * CELL_VOLUME

def total_enstrophy(u, T):
    mu  = dynamic_viscosity(u, T)
    w   = mu * sum(vorticity_3d(u, T)**2, axis = 0)
    return sum(w) * CELL_VOLUME

def total_entropy(u, T):
    rho = u[0]
    s = - rho * entropy(rho, T)
    return sum(s) * CELL_VOLUME

def total_kinetic_energy(u, T):
    rho = u[0]
    k = kinetic_energy(rho, u[1:(N_DIMENSIONS+1)] / rho)
    return sum(k) * CELL_VOLUME


if N_DIMENSIONS == 2:
    vorticity = vorticity_2d
elif N_DIMENSIONS == 3:
    vorticity = vorticity_3d
