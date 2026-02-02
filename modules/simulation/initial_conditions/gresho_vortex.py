"""
Docstring for modules.simulation.initial_conditions.gresho_vortex
"""

from prep_jax import *
from config.conf_geometry       import *
from config.conf_thermodynamics import *

from jax.numpy                      import where, logical_and, sqrt, ones_like, log, stack, array, exp
from jax.scipy.special              import expi
from jax                            import jit
from modules.numerical.computation  import pad_1d_to_mesh, extract_1d_from_padded
from modules.thermodynamics.EOS     import speed_of_sound, temperature_rpt, pressure

from modules.thermodynamics.gas_models.ideal_gas import temperature_rpt_ideal

rho_c, T_c, p_c = molecule.critical_point

def _V_shape(r_tilde):
    vel = r_tilde * (1 - r_tilde)**2
    return where(r_tilde <= 1, vel, 0)

def _V_shape_integral(r_tilde):
    r = where(r_tilde <= 1, r_tilde, 1)
    return 1/2 * r**2 - 4/3 * r**3 + 3/2 * r**4 - 4/5 * r**5 + 1/6 * r**6

#@jit
def gresho_vortex(mesh):
    """
    Domain [0, 2] x [0, 1]
    """
    
    x_c = 1
    y_c = 0.5

    r = sqrt((mesh[0] - x_c)**2 + (mesh[1] - y_c)**2)
    R = 0.25
    V_ref = 0.1

    Vr = V_ref * _V_shape(r / R)

    du = Vr * (-(mesh[1] - y_c)) / r 
    dv = Vr * (mesh[0] - x_c) / r 

    rho_inf = 1.1 * rho_c
    T_inf   = 1.1 * T_c
    
    p_inf   = extract_1d_from_padded(
        pressure(
            pad_1d_to_mesh(array([rho_inf])),
            pad_1d_to_mesh(array([T_inf]))
        )
    )[0]

    c       = extract_1d_from_padded(
        speed_of_sound(
            pad_1d_to_mesh(array([rho_inf])),
            pad_1d_to_mesh(array([T_inf]))
        )
    )[0]

    U_inf = 0.01 * c

    print('U_inf:', U_inf)

    rho = rho_inf * ones_like(mesh[0])
    u   = U_inf + du
    v   = dv
    p   = p_inf - rho_inf * V_ref**2 * (1 / 30 - _V_shape_integral(r / R)) 


    return stack((rho, u, v, p)), 0




    


