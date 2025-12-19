"""
    Implementation of the dynamic viscosity as prescribed by Laesecke and Muzny (2017)

    ``Reference Correlation for the Viscosity of Carbon Dioxide''
"""

from prep_jax import *
from config.conf_thermodynamics import *

''' Parameters '''

#hard-coded
zero_density_lim_coeffs = jnp.array([
    1749.354893188350,
    -369.069300007128,
    5423856.34887691,
    -2.21283852168356,
    -269503.247933569,
    73145.021531826,
    5.34368649509278,
])

virial_coeffs = jnp.array([
    -19.572881,
    219.73999,
    -1015.3226,
    2471.0125,
    -3375.1717,
    2491.6597,
    -787.26086,
    14.085455,
    -0.34664158
])

virial_exponents = jnp.array([
    0.25, 
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.5,
    5.5
])

energy_scaling_parameter = 200.75967461 #K
#length_scaling_parameter = 0.378421 #nm
sigma3Na = 3.2634491775779287e-5 #m^3 mol^-1
dimensioning_factor = 0.0943605819425392 #mPa sec
gamma_coeff = 8.06282737481277
c1_coeff = 0.360603235428487
c2_coeff = 0.121550806591497


''' Functions'''

def check_consistency():
    """
    Docstring for check_consistency
    """
    assert NAME_MOLECULE == "CO_2", "Laesecke dynamic viscosity model is only valid for carbon dioxide" 


def laesecke_zero_density_limit(u, T):
    """
        returns dynamic viscosity in mPa sec in limit of zero density
    """
    T_sqrt = jnp.sqrt(T)
    T_one_third = jnp.pow(T, 1/3)

    num = 1.0055 * T_sqrt
    denom = zero_density_lim_coeffs[0] \
    + zero_density_lim_coeffs[1] * jnp.pow(T, 1/6) \
    + zero_density_lim_coeffs[2] * jnp.exp(zero_density_lim_coeffs[3] * T_one_third) \
    + (zero_density_lim_coeffs[4] + zero_density_lim_coeffs[5] * T_one_third) / jnp.exp(T_one_third) \
    + zero_density_lim_coeffs[6] * T_sqrt

    return num / denom 
    

def laesecke_linear_in_density(u, T):
    """
        returns the linear-in-density contribution to dynamic viscosity in mPa sec
    """
    eta_0 = laesecke_zero_density_limit(u, T)

    T_red = T / energy_scaling_parameter 

    virial_term = virial_coeffs[0] * jnp.ones_like(T)
    for i in range(8):
        virial_term += virial_coeffs[i+1] / jnp.pow(T_red, virial_exponents[i])

    return eta_0 * virial_term * sigma3Na / MOLAR_MASS * u[0] #check units

def laesecke_residual_term(u, T):
    """
        returns the residual contribution to the dynamcic viscosity in mPa sec
    """

    T_red = T / 216.592 #hard-coded triple point temperature -> refprop value
    rho_red = u[0] / 1178.53 #hard-coded liquid density at triple point -> Laesecke paper

    return dimensioning_factor * ( \
        c1_coeff * T_red * rho_red**3 \
        + (rho_red**2 + rho_red ** gamma_coeff) / (T_red - c2_coeff)
    )

def laesecke_dynamic_viscosity(u, T):
    """
        Computes the dynamic viscosity in Pa sec according to Laesecke and Muzny (2017)
    """
    # return in Pa sec
    return 1e-3 * (laesecke_zero_density_limit(u, T) + laesecke_linear_in_density(u, T) + laesecke_residual_term(u, T))
