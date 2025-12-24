"""
    Functions for thermodynamic equations of state (EOS).
"""

from prep_jax import *
from config.conf_thermodynamics import *
from config.conf_geometry import N_DIMENSIONS

''' Consistency checks '''

KNOWN_EOS = ["IDEAL_GAS", "VAN_DER_WAALS", "PENG_ROBINSON", "WAGNER"]

assert EOS in KNOWN_EOS, f"Unknown EOS: {EOS}"
assert MOLAR_MASS > 0, f"Molar mass should be positive"

match EOS:
    case "IDEAL_GAS":

        from modules.thermodynamics.gas_models.ideal_gas import check_consistency_ideal as check_consistency

        ''' set critical point values '''
        #from modules.thermodynamics.gas_models.ideal_gas import rho_c, T_c, p_c

        '''' set Helmholtz energy function for ideal gas '''
        from modules.thermodynamics.gas_models.ideal_gas import ideal_gas as Helmholtz

        ''' set temperature function for ideal gas '''
        from modules.thermodynamics.gas_models.ideal_gas import temperature_rpt_ideal as temperature_rpt

        ''' set density function for ideal gas '''
        from modules.thermodynamics.gas_models.ideal_gas import density_ptr_ideal as density_ptr

        ''' set temperature functions for ideal gas '''
        from modules.thermodynamics.gas_models.ideal_gas import temperature_ret_ideal as temperature_ret


    case "VAN_DER_WAALS":

        from modules.thermodynamics.gas_models.Van_der_Waals import check_consistency_Van_der_Waals as check_consistency

        ''' set critical point values '''
        #from modules.thermodynamics.gas_models.Van_der_Waals import rho_c, T_c, p_c

        ''' set Helmholtz energy function for Van der Waals '''
        from modules.thermodynamics.gas_models.Van_der_Waals import Van_der_Waals as Helmholtz

        ''' set temperature function for Van der Waals gas '''        
        from modules.thermodynamics.gas_models.Van_der_Waals import temperature_rpt_Van_der_Waals as temperature_rpt

        ''' set density function for Van der Waals gas '''
        from modules.thermodynamics.gas_models.Van_der_Waals import density_ptr_Van_der_Waals as density_ptr

        ''' set temperature functions for Van der Waals gas '''
        from modules.thermodynamics.gas_models.Van_der_Waals import temperature_ret_Van_der_Waals as temperature_ret


    case "PENG_ROBINSON":

        from modules.thermodynamics.gas_models.Peng_Robinson import check_consistency_Peng_Robinson as check_consistency

        ''' set critical point values '''
        #from modules.thermodynamics.gas_models.Peng_Robinson import rho_c, T_c, p_c

        ''' set Helmholtz energy function for Span_Wagner '''
        from modules.thermodynamics.gas_models.Peng_Robinson import Peng_Robinson as Helmholtz

        ''' set temperature function for Span_Wagner gas '''        
        from modules.thermodynamics.gas_models.Peng_Robinson import temperature_rpt_Peng_Robinson as temperature_rpt

        ''' set density function for Span_Wagner gas '''
        from modules.thermodynamics.gas_models.Peng_Robinson import density_ptr_Peng_Robinson as density_ptr

        ''' set temperature functions for Span_Wagner gas '''
        from modules.thermodynamics.gas_models.Peng_Robinson import temperature_ret_Peng_Robinson as temperature_ret


    case "WAGNER":

        from modules.thermodynamics.gas_models.Wagner import check_consistency_Wagner as check_consistency

        ''' set critical point values '''
        #from modules.thermodynamics.gas_models.Wagner import rho_c, T_c, p_c

        ''' set Helmholtz energy function for Wagner '''
        from modules.thermodynamics.gas_models.Wagner import Wagner as Helmholtz

        ''' set temperature function for Wagner gas '''        
        from modules.thermodynamics.gas_models.Wagner import temperature_rpt_Wagner as temperature_rpt

        ''' set density function for Wagner gas '''
        from modules.thermodynamics.gas_models.Wagner import density_ptr_Wagner as density_ptr

        ''' set temperature functions for Wagner gas '''
        from modules.thermodynamics.gas_models.Wagner import temperature_ret_Wagner as temperature_ret

    case _:
        raise ValueError(f"Unknown EOS: {EOS}")

check_consistency()


''' Temperature equations (rho, e) -> T for simulations '''
def internal_energy_u(u):
    """
        Calculate internal energy from conservative variables
    """
    return u[N_DIMENSIONS+1] / u[0] - 0.5 * jnp.sum((u[1:N_DIMENSIONS+1])**2, axis=0) / u[0]**2


''' Temperature from conservative variables  (u) -> T '''
temperature = lambda u, Tguess: temperature_ret(u[0], internal_energy_u(u), Tguess)



''' Compute other thermodynamic quantities automatically '''
def _vectorize_thermo(f : callable) -> callable:
    """
    Helper function to clean up code

    Vectorizes a thermodynamic function to work on arrays of dim = N_DIMENSIONS 
    elementwise
    """
    f_vec = jax.vmap(f, in_axes=(0, 0), out_axes=0)
    for i in range(1, N_DIMENSIONS):
        f_vec = jax.vmap(f_vec, in_axes=(i, i), out_axes=i)
    
    return f_vec


''' Helmholtz and derivatives mapped over n dimensions '''
dAdrho_scalar       = jax.grad(Helmholtz, argnums = 0)
dAdT_scalar         = jax.grad(Helmholtz, argnums = 1)
Helmholtz_scalar    = Helmholtz

dAdrho      = _vectorize_thermo(dAdrho_scalar)
dAdT        = _vectorize_thermo(dAdT_scalar)
Helmholtz   = _vectorize_thermo(Helmholtz_scalar)

''' Helmholtz second derivatives mapped over n dimensions '''
d2Ad2rho_scalar     = jax.grad(dAdrho_scalar, argnums = 0)
d2Ad2T_scalar       = jax.grad(dAdT_scalar, argnums = 1)
d2AdrhodT_scalar    = jax.grad(dAdrho_scalar, argnums=1)

d2Ad2rho    = _vectorize_thermo(d2Ad2rho_scalar)
d2Ad2T      = _vectorize_thermo(d2Ad2T_scalar)
d2AdrhodT   = _vectorize_thermo(d2AdrhodT_scalar)

''' Helmholtz third derivatives mapped over n dimensions'''
d3Ad3rho_scalar     = jax.grad(d2Ad2rho_scalar, argnums = 0)
d3Ad2rhodT_scalar   = jax.grad(d2Ad2rho_scalar, argnums = 1)
d3Ad2Tdrho_scalar   = jax.grad(d2Ad2T_scalar, argnums = 0)
d3Ad3T_scalar       = jax.grad(d2Ad2T_scalar, argnums = 1)

d3Ad3rho    = _vectorize_thermo(d3Ad3rho_scalar)
d3Ad2rhodT  = _vectorize_thermo(d3Ad2rhodT_scalar)
d3Ad2Tdrho  = _vectorize_thermo(d3Ad2Tdrho_scalar)
d3Ad3T      = _vectorize_thermo(d3Ad3T_scalar)


''' Computing thermodynamic quantites using the equation of state'''
def pressure(rho, T):
    return rho**2 * dAdrho(rho, T) 

''' all thermodynamic quantities below are defined per unit mass '''
def entropy(rho, T):
    return -dAdT(rho, T)

def internal_energy(rho, T):
    return Helmholtz(rho, T) - T * dAdT(rho, T)

def Gibbs_energy(rho, T):
    return Helmholtz(rho, T) + rho * dAdrho(rho, T) 

def enthalpy(rho, T):
    return Helmholtz(rho, T) + rho * dAdrho(rho, T) - T * dAdT(rho, T)


''' all thermodynamic quantities below are defined per unit volume '''
def kinetic_energy(rho, v):
    return 0.5 * rho * jnp.sum(v**2, axis = 0)

def total_energy(rho, T, v):
    return rho * internal_energy(rho, T) + kinetic_energy(rho, v)


''' miscellaneous thermodynamic quantities'''
def speed_of_sound(rho, T):
    """
        IMPORTANT: NOT SQUARED
    """
    s = 2 * rho * dAdrho(rho, T) + rho**2 * d2Ad2rho(rho, T) - rho**2 * (d2AdrhodT(rho,T)**2) / d2Ad2T(rho,T)
    return jnp.sqrt(jnp.abs(s))

def c_v(rho, T):
    """
        Specific isochoric heat capacity c_v.
    """
    A_TT = d2Ad2T(rho, T)
    c_v = -T * A_TT
    return c_v

def c_p(rho, T):
    """
        Specific isobaric heat capacity c_p.
    """
    A_rho      = dAdrho(rho, T)
    A_rhorho   = d2Ad2rho(rho, T)
    A_TT       = d2Ad2T(rho, T)
    A_rhoT     = d2AdrhodT(rho, T)

    c_v = -T * A_TT
    dp_drho_T = 2 * rho * A_rho + rho**2 * A_rhorho     
    dp_dT_rho = rho**2 * A_rhoT                        

    cp = c_v + T * (dp_dT_rho**2) / (rho**2 * dp_drho_T)
    return cp


''' all thermodynamic quantities below are used specifically in numerical fluxes'''
# keepdg terms
#-----------
def Gibbs_beta(rho, beta):
    return beta * (Helmholtz_scalar(rho, 1/beta) + rho * dAdrho_scalar(rho, 1/beta) )

dgbdrho = jax.grad(Gibbs_beta, argnums = 0)
dgbdbeta = jax.grad(Gibbs_beta, argnums = 1)

Gibbs_beta = _vectorize_thermo(Gibbs_beta)
dgbdrho = _vectorize_thermo(dgbdrho)
dgbdbeta = _vectorize_thermo(dgbdbeta)

def pressure_beta(rho, beta):
    return beta * (rho**2 * dAdrho_scalar(rho, 1/beta))

dpbdrho = jax.grad(pressure_beta, argnums = 0)
dpbdbeta = jax.grad(pressure_beta, argnums = 1)

pressure_beta = _vectorize_thermo(pressure_beta)
dpbdrho = _vectorize_thermo(dpbdrho)
dpbdbeta = _vectorize_thermo(dpbdbeta)

''' derivatives used in manufactured solutions '''
pressure_rho = jax.grad(lambda rho, T: rho**2 * dAdrho_scalar(rho, T), 0)
pressure_T   = jax.grad(lambda rho, T: rho**2 * dAdrho_scalar(rho, T), 1)

pressure_rho = _vectorize_thermo(pressure_rho)
pressure_T   = _vectorize_thermo(pressure_T) 

internal_energy_rho = jax.grad(lambda rho, T: Helmholtz_scalar(rho, T) - T * dAdT_scalar(rho, T), 0)
internal_energy_T   = jax.grad(lambda rho, T: Helmholtz_scalar(rho, T) - T * dAdT_scalar(rho, T), 1)

internal_energy_rho = _vectorize_thermo(internal_energy_rho)
internal_energy_T   = _vectorize_thermo(internal_energy_T) 