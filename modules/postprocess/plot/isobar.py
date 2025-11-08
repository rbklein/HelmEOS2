'''
    Compute the density and temperature values associated to some isobar
'''
from prep_jax import *
from modules.thermodynamics.EOS import *
#from modules.thermodynamics.gas_models.ideal_gas import temperature_rpt_ideal, density_ptr_ideal
from modules.thermodynamics.gas_models.Van_der_Waals import temperature_rpt_Van_der_Waals
from modules.thermodynamics.gas_models.Peng_Robinson import density_ptr_Peng_Robinson
from modules.numerical.computation import pad_1d_to_mesh, extract_1d_from_padded

def isobar_rho(p_iso, rho1, rho2, num_points = 100):
    rho = pad_1d_to_mesh(jnp.linspace(rho1, rho2, num_points))
    p = p_iso * jnp.ones_like(rho)
    T = temperature_rpt_Van_der_Waals(rho, p, None)
    T = temperature_rpt(rho, p, T)
    return extract_1d_from_padded(rho), extract_1d_from_padded(T)

def isobar_T(p_iso, T1, T2, num_points = 100):
    T = pad_1d_to_mesh(jnp.linspace(T1, T2, num_points))
    p = p_iso * jnp.ones_like(T)
    rho = density_ptr_Peng_Robinson(p, T, None)
    rho = density_ptr(p, T, rho)
    return extract_1d_from_padded(rho), extract_1d_from_padded(T)