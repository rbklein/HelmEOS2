"""
The Helmholtz energy and other functions of the Peng-Robinson equation
"""

from prep_jax import *
from config.conf_thermodynamics import *
from config.conf_geometry import *

''' check parameter consistency '''

def check_consistency_Peng_Robinson():
    assert "acentric_factor" in EOS_parameters, "EOS parameters for Van der Waals must include 'acentric_factor'"
    assert "molecular_dofs" in EOS_parameters, "EOS parameters for Van der Waals must include 'molecular_dofs'"
    assert isinstance(EOS_parameters["acentric_factor"], float), "EOS parameter 'acentric_factor' must be a number"
    assert isinstance(EOS_parameters["molecular_dofs"], int), "EOS parameter 'molecular_dofs' must be a number"
    assert EOS_parameters["molecular_dofs"] > 0, "EOS parameter 'molecular_dofs' must be greater than 0"

''' Derived parameters '''

R_specific = UNIVERSAL_GAS_CONSTANT / MOLAR_MASS #J K^-1 kg^-1 specific gas constant

''' set parameter values for Peng-Robinson '''

rho_c, T_c, p_c = molecule.critical_points

# reported in: Fully Compressible Low-Mach Number Simulations of Carbon-dioxide at Supercritical Pressures and Trans-critical Temperatures, Sengupta et al.
a_PR = 0.457235 * (R_specific * T_c**2) / p_c
b_PR = 0.077796 * (R_specific * T_c) / p_c
kappa = 0.37464 + 1.54226 * molecule.Peng_Robinson_parameters["acentric_factor"] - 0.26992 * molecule.Peng_Robinson_parameters["acentric_factor"]**2
molecular_dofs = molecule.Peng_Robinson_parameters["molecular_dofs"]

''' Helmholtz energy '''

def attraction_func(T):
    """
        The temperature-dependent attraction function in PR
    """
    return a_PR * (1 + kappa * (1 - jnp.sqrt(T / T_c)))**2

def Peng_Robinson(rho, T):
    """
        Specific Helmholtz energy of the Peng-Robinson equation of state
    """
    alpha = attraction_func(T)
    density_term = (1 + (1+jnp.sqrt(2)) * b_PR * rho) / (1 + (1-jnp.sqrt(2)) * b_PR * rho)

    return - R_specific * T * (1 + jnp.log((1-rho*b_PR) * T**(molecular_dofs / 2) / rho)) - alpha / (2 * jnp.sqrt(2) * b_PR) * jnp.log(density_term)

''' Temperature equation (rho, p) -> T for initial conditions'''
@jax.jit
def _root_func_pressure(rho, p, T):
    """
        function f(T) = p - p(rho,T)  to be solved for root
    """
    res = p - (rho * R_specific * T) / (1 - rho * b_PR) + (attraction_func(T) * rho**2) / (1 + 2 * b_PR * rho - b_PR**2 * rho**2)
    return res

_drpdT_root = jax.grad(_root_func_pressure, 2,)

@jax.jit
def _solve_root_pressure(rho, p):
    #initial guess, exact solution ignoring molecular attraction
    T = p * (1 - b_PR * rho) / (rho * R_specific)
    it_max = 10

    def cond(state):
        T, i = state
        return jnp.logical_and(i < it_max, jnp.abs(_root_func_pressure(rho, p, T)))
    
    def body(state):
        T, i = state
        res = _root_func_pressure(rho, p, T)
        dres = _drpdT_root(rho, p, T)
        dres = jnp.where(jnp.abs(dres) < 1e-20, jnp.sign(dres) * 1e-20, dres)
        step = res / dres
        T = T - jnp.clip(step, -10., 10.)
        return (T, i + 1)
    
    return jax.lax.while_loop(cond, body, (T, jnp.array(0)))[0]

for i in range(N_DIMENSIONS):
    _solve_root_pressure = jax.vmap(_solve_root_pressure, (i, i), i)

def temperature_eos_Peng_Robinson(rho, p):
    """
        Solve temperature profile from density and pressure for Peng-Robinson gas
    """
    return _solve_root_pressure(rho, p)

''' Density equation (p, T) -> rho for initial conditions'''

from modules.numerical.computation import cubic_real_roots
def density_eos_Peng_Robinson(p, T):
    """
        Solve the cubic compressibility equation for Z and recover density

        The equation is given in the original PR paper:
        'A new Two-Constant Equation of State' by Peng and Robinson
    """
    #compute PR's A, B parameters
    A = (a_PR * p) / (R_specific**2 * T**2)
    B = (b_PR * p) / (R_specific * T)

    #compute coefficients
    p3 = jnp.ones_like(T)
    p2 = -(1-B)
    p1 = A - 3*B**2 - 2*B
    p0 = -(A * B - B**2 - B**3)

    #compute compressibility factor
    coeffs = jnp.stack((p3, p2, p1, p0), axis = 0)
    r, mask, num_real = cubic_real_roots(coeffs)
    Z = r[..., 0]

    #compute density from compressibility factor
    rho = p / (Z * R_specific * T)
    return rho


''' Temperature equations (rho, e) -> T for simulations '''

grad_attraction = jax.grad(attraction_func)

@jax.jit
def _root_func_energy(rho, e, T):
    """
        function f(T) = e - e(rho,T)  to be solved for root
    """

    num = grad_attraction(T) * T - attraction_func(T)
    density_term = (1 + (1+jnp.sqrt(2)) * b_PR * rho) / (1 + (1-jnp.sqrt(2)) * b_PR * rho)

    return e - molecular_dofs / 2 * R_specific * T - num / (2 * jnp.sqrt(2) * b_PR) * jnp.log(density_term)

_dredT_root = jax.grad(_root_func_energy, 2)

@jax.jit
def _solve_root_energy(rho, e):
    #initial guess, exact solution ignoring molecular attraction
    T = e / (molecular_dofs / 2 * R_specific)
    it_max = 10

    def cond(state):
        T, i = state
        return jnp.logical_and(i < it_max, jnp.abs(_root_func_energy(rho, e, T)))
    
    def body(state):
        T, i = state
        res = _root_func_energy(rho, e, T)
        dres = _dredT_root(rho, e, T)
        dres = jnp.where(jnp.abs(dres) < 1e-20, jnp.sign(dres) * 1e-20, dres)
        step = res / dres
        T = T - jnp.clip(step, -10., 10.)
        return (T, i + 1)
    
    return jax.lax.while_loop(cond, body, (T, jnp.array(0)))[0]

for i in range(N_DIMENSIONS):
    _solve_root_energy = jax.vmap(_solve_root_energy, (i, i), i)

def temperature_Peng_Robinson(rho, e):
    """
        Calculate temperature from density and specific internal energy for Peng-Robinson EOS.
    """
    return _solve_root_energy(rho, e)



