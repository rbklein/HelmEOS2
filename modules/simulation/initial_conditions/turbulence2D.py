"""
    Contains functions to generate initial conditions for 2D compressible turbulence simulations.

    generated with ChatGPT 5
"""

from prep_jax import *
from modules.geometry.grid import *
from jax.scipy.special import gamma
from typing import Tuple

from modules.thermodynamics.EOS import *

''' Consistency checks '''

# initial kinetic energy spectrum 
# should be configured so that max(E(kxmax),E(kymax)) < epsilon
# so that energy in corners of kx, ky space is negligible
def E(k : float | jnp.ndarray, shape : float = 3, k_peak : float = 12.0) -> float | jnp.ndarray:
    """
        Model for initial kinetic energy spectrum of a 2D velocity field

        Input:
            k: wavenumber magnitude i.e. sqrt(kx^2 + ky^2)
            shape: shape factor, larger values center energy at k_peak
            k_peak: wavenumber with maximum energy
        Output:
            E(k): energy spectrum at wavenumber magnitude k
    """
    factorial = lambda n: gamma(n + 1)
    normalization = (2 * shape + 1)**(shape + 1) / (2**shape * factorial(shape))

    exponent = - (shape + 1/2) * (k / k_peak)**2
    factor = normalization / 2 * 1 / k_peak * (k / k_peak)**(2 * shape + 1)
    E_k = factor * jnp.exp(exponent)

    return jnp.maximum(E_k, 1e-14)

def random_velocity(key : jax.random.KeyArray, E : jnp.ndarray) -> jnp.ndarray:

    dx, dy = GRID_SPACING

    #split random key
    key_phase_x, key_phase_y = jax.random.split(key, 2)

    #determine fraction of energy between ux and uy
    coeffs_abs = 2 * E / 4                      # / 4 to account for symmetry in fourier space (this way stuff that is aliased to some wavenumber all contribute equal to that wavenumber)
    coeffs_sq_abs_x = 1/2 * coeffs_abs
    coeffs_sq_abs_y = 1/2 * coeffs_abs

    #find random phases of real signal by taking phase of random real signal
    x_r = jax.random.uniform(key_phase_x, shape=coeffs_sq_abs_x.shape, minval=0.0, maxval = 1.0)
    c_x_r = jnp.fft.fft2(x_r) / (dx * dy)
    rand_phase_x = jnp.angle(c_x_r) + jnp.pi

    y_r = jax.random.uniform(key_phase_y, shape=coeffs_sq_abs_x.shape, minval=0.0, maxval = 1.0)
    c_y_r = jnp.fft.fft2(y_r) / (dx * dy)
    rand_phase_y = jnp.angle(c_y_r) + jnp.pi

    #determine fourier coefficients with random phases
    coeffs_x = jnp.sqrt(coeffs_sq_abs_x) * jnp.exp(1j * rand_phase_x)
    coeffs_y = jnp.sqrt(coeffs_sq_abs_y) * jnp.exp(1j * rand_phase_y)

    ux = jnp.fft.ifft2(coeffs_x) / (dx * dy)
    uy = jnp.fft.ifft2(coeffs_y) / (dx * dy)
    return ux.real, uy.real

def random_velocity_solenoidal(key: jax.random.KeyArray, E: jnp.ndarray):
    """
    Generate a 2D approximately solenoidal velocity field (ux, uy)
    with a prescribed per-mode energy spectrum E (same shape as the grid).
    """
    dx, dy = GRID_SPACING
    nx, ny = E.shape

    # Wavenumbers (radians per unit length)
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    # Target |û|^2 per mode. Your original factor keeps isotropic symmetry handling.
    coeffs_abs = 2.0 * E / 4.0  # == |û|^2 per (kx, ky)

    # Random phases from the FFT of a random real field (guarantees Hermitian symmetry)
    psi_r = jax.random.uniform(key, shape=E.shape, minval=0.0, maxval=1.0)
    cpsi = jnp.fft.fft2(psi_r) / (dx * dy)
    theta = jnp.angle(cpsi) + jnp.pi

    # Set |ψ̂| so that |û|^2 = |k|^2 |ψ̂|^2 matches coeffs_abs
    abs_psi = jnp.where(K2 > 0.0, jnp.sqrt(coeffs_abs) / jnp.sqrt(K2), 0.0)
    psi_hat = abs_psi * jnp.exp(1j * theta)

    # û = i k⊥ ψ̂ = i * (-ky, kx) * ψ̂
    uhat_x = 1j * (-KY) * psi_hat
    uhat_y = 1j * ( KX) * psi_hat

    # Back to real space (keep same normalization convention as your code)
    ux = jnp.fft.ifft2(uhat_x) / (dx * dy)
    uy = jnp.fft.ifft2(uhat_y) / (dx * dy)

    return ux.real, uy.real

rho_c, T_c, p_c = molecule.critical_points

def turbulence_2d(mesh : Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
        Generate a 2D turbulence initial condition.
    """

    dx, dy = GRID_SPACING
    nx, ny = GRID_RESOLUTION

    # wavenumber coordinates
    kx = jnp.fft.fftfreq(nx, d=dx)
    ky = jnp.fft.fftfreq(ny, d=dy)
    Kx, Ky = jnp.meshgrid(kx, ky, indexing = 'ij')
    K_mag = jnp.sqrt(Kx**2 + Ky**2)

    Ek =  E(K_mag, shape=3, k_peak = 8)
    key = jax.random.PRNGKey(0)
    ux, uy = random_velocity_solenoidal(key, Ek)

    rho =  1.5 * rho_c * jnp.ones_like(mesh[0])  # Uniform density field
    p = 2.5 * p_c * jnp.ones_like(mesh[0])  # Uniform pressure field
    return jnp.stack((rho, ux, uy, p), axis=0), 0 #rvp