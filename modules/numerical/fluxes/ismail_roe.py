from prep_jax import *
from config.conf_numerical      import *
from config.conf_thermodynamics import *

from modules.geometry.grid          import GRID_RESOLUTION, GRID_SPACING
from modules.thermodynamics.EOS     import pressure, sqrt
from modules.numerical.computation  import log_mean
from jax.numpy                      import stack


_gamma = molecule.ideal_gas_parameters['gamma']

def div_ismail_roe_1d(u, T):
    #compute necessary fields
    density     = u[0] 
    temp        = T
    vel         = u[1] / density
    press       = pressure(density, temp)

    z1 = sqrt(density / press)
    z2 = z1 * vel
    z3 = sqrt(density * press)

    z1_mean = (z1[1:] + z1[:-1]) / 2
    z1_log  = log_mean(z1[1:], z1[:-1])
    z2_mean = (z2[1:] + z2[:-1]) / 2
    z3_log  = log_mean(z3[1:], z3[:-1])
    z3_mean = (z3[1:] + z3[:-1]) / 2

    gamma = _gamma

    v = z2_mean / z1_mean
    rho = z1_mean * z3_log
    p1  = z3_mean / z1_mean
    p2  = (gamma + 1) / (2 * gamma) * z3_log / z1_log + (gamma - 1) / (2 * gamma) * z3_mean / z1_mean
    a   = sqrt(gamma * p2 / rho)
    H   = a**2 / (gamma - 1) + 0.5 * v**2

    f_density = rho * v
    f_momentum = rho * v**2 + p1
    f_energy = rho * v * H

    f = stack([f_density, f_momentum, f_energy], axis=0)
    return f[:, 1:] - f[:, :-1]