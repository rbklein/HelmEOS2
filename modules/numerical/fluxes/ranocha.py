from prep_jax import *
from config.conf_numerical      import *
from config.conf_thermodynamics import *

from modules.geometry.grid          import GRID_RESOLUTION, GRID_SPACING
from modules.thermodynamics.EOS     import pressure
from modules.numerical.computation  import log_mean
from jax.numpy                      import stack


_gamma = molecule.ideal_gas_parameters['gamma']

def div_ranocha_1d(u, T):
    """
    Assume u is padded appropriately (5, n_x + 2, n_y + 2).
    """
    """
    Assume u is padded appropriately (5, n_x + 2, n_y + 2).
    """

    #compute necessary fields
    density     = u[0] 
    temp        = T
    vel         = u[1] / density
    press       = pressure(density, temp)

    density_mean = log_mean(density[1:], density[:-1])
    vel_mean    = 0.5 * (vel[:-1] + vel[1:])
    f_density   = density_mean * vel_mean

    #compute pressure in x-direction
    pressure_mean   = 0.5 * (press[:-1] + press[1:]) 
    f_momentum      = f_density * vel_mean + pressure_mean


    gamma           = _gamma
    vel_prod_mean   = 2 * vel_mean**2 - 0.5 * (vel[:-1]**2 + vel[1:]**2)
    rho_p           = density / press
    rho_p_mean      = log_mean(rho_p[1:], rho_p[:-1])
    pv              = press * vel
    pv_mean         = 2 * pressure_mean * vel_mean - 0.5 * (pv[:-1] + pv[1:])

    f_internal_energy   = f_density / rho_p_mean / (gamma - 1)
    f_kinetic_energy    = f_density * 1/2 * vel_prod_mean
    f_pv                = pv_mean    

    f_total_energy = f_internal_energy + f_kinetic_energy + f_pv
    f = stack((f_density, f_momentum, f_total_energy), axis=0)

    return f[:, 1:] - f[:, :-1]


    
