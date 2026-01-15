from prep_jax import *
from config.conf_numerical      import *
from config.conf_thermodynamics import *

from modules.geometry.grid          import GRID_RESOLUTION, GRID_SPACING
from modules.numerical.computation  import log_mean
from jax.numpy                      import stack


_gamma = molecule.ideal_gas_parameters['gamma']

def div_chandrashekar_1d(u, T):
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

    rho_log = log_mean(density[1:], density[:-1])
    vel_mean    = 0.5 * (vel[:-1] + vel[1:])

    f_density   = rho_log * vel_mean    

    beta = 1 / T

    rho_mean    = (density[1:] + density[:-1]) / 2
    beta_mean   = (beta[1:] + beta[:-1]) / 2
    p_tilde     = rho_mean / (2 * beta_mean)

    f_momentum  = f_density * vel_mean + p_tilde

    gamma       = _gamma
    beta_log    = log_mean(beta[1:], beta[:-1])
    u2_mean     = (vel[:-1]**2 + vel[1:]**2) / 2

    f_total_energy    = (1 / (2 * (gamma - 1) * beta_log) - 0.5 * u2_mean) * f_density + vel_mean * f_momentum

    f = stack((f_density, f_momentum, f_total_energy), axis=0)
    return f[:, 1:] - f[:, :-1]
