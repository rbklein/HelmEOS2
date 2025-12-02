"""
Plot p-T, p-v/rho and rho-T diagrams
"""


from prep_jax import *
from modules.thermodynamics.EOS import *
from modules.postprocess.plot.isobar    import isobar_rho, isobar_T
from modules.numerical.computation      import pad_1d_to_mesh, extract_1d_from_padded

rho_c, T_c, p_c = molecule.critical_point

def plot_pT(fig, ax):
    n_isochors = 10
    resolution = 100

    idx = (resolution * 99) // 100

    isochors = rho_c * jnp.linspace(1.0, 2.0, n_isochors)

    T_range = T_c * jnp.linspace(1.0, 2.0, resolution)
    T_pad   = pad_1d_to_mesh(T_range)
    
    for isochor in isochors:
        p = pressure(isochor * jnp.ones_like(T_pad), T_pad)
        p = extract_1d_from_padded(p)
        
        ax.loglog(T_range / T_c, p / p_c)
        ax.text(T_range[idx] / T_c, p[idx] / p_c, f"{(isochor / rho_c):g}", fontsize=9, va="center", ha="left")

    ax.set_xlabel(r'$T_r$')
    ax.set_ylabel(r'$p_r$')
    ax.grid(which = 'both')



def plot_pv(fig, ax, plot_density = False):
    n_isotherms = 10
    resolution = 100

    idx = (resolution * 99) // 100

    isotherms = jnp.linspace(1.0 * T_c, 2.0 * T_c, n_isotherms)

    rho_range = jnp.linspace(0.1 * rho_c, 2.0 * rho_c, resolution)
    rho_pad = pad_1d_to_mesh(rho_range)

    for isotherm in isotherms:
        p = pressure(rho_pad, isotherm * jnp.ones_like(rho_pad))
        p = extract_1d_from_padded(p)

        if not plot_density:
            ax.loglog(rho_c / rho_range, p / p_c)
            ax.text(rho_c / rho_range[idx], p[idx] / p_c, f"{(isotherm / T_c):g}", fontsize=9, va="center", ha="left")
        else:
            ax.loglog(rho_range / rho_c, p / p_c)
            ax.text(rho_range[idx] / rho_c, p[idx] / p_c, f"{(isotherm / T_c):g}", fontsize=9, va="center", ha="left")

    if not plot_density:
        ax.set_xlabel(r'$v_r$')
    else:
        ax.set_xlabel(r'$\rho_r$')

    ax.set_ylabel(r'$p_r$')
    ax.grid(which = 'both')


def plot_rhoT(fig, ax):
    n_isobars  = 10
    resolution = 100

    idx = (resolution * 99) // 100

    isobars = p_c * jnp.linspace(1.1, 2.0, n_isobars)

    T_range = jnp.linspace(T_c, 2 * T_c, resolution)
    T_pad   = pad_1d_to_mesh(T_range)

    for isobar in isobars:
        rho, T = isobar_T(isobar, T_range[0], T_range[resolution-1], resolution)

        ax.loglog(rho / rho_c, T / T_c) 
        ax.text(rho[idx] / rho_c, T[idx] / T_c, f"{(isobar / p_c):g}", fontsize=9, va="center", ha="left")

    ax.set_xlabel(r'$\rho_r$')
    ax.set_ylabel(r'$T_r$')
    ax.grid(which = 'both')
