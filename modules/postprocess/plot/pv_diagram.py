"""
functions to compute a pv-diagram with liquid vapor equilibrium region
"""

'''
from prep_jax import *
from modules.thermodynamics.EOS import *
from modules.numerical.computation import convex_envelope

#TO DO: return NaN if no solution exists (e.g. for ideal gas)
def solve_VLE_pressure_in_interval(T, rho1 = 0.1, rho2 = 1):
    """
        Compute the VLE interval for a given isotherm by computing the lower convex envelope of the helmholtz energy as a function of 
        specific volume 'v' and determining for what v it differs from the helmholtz energy given by the equation of state, see 
        Markus Deserno "Van der Waals equation, Maxwell construction, and Legendre transforms"

        values are reduced

    """
    v       = jnp.logspace(jnp.log10(1/rho2), jnp.log10(1/rho1), 1000)
    helm    = Helmholtz_scalar(1/v * rho_c, T * T_c)

    envelope_indices = convex_envelope(v, helm)

    index_v1 = jnp.nonzero((envelope_indices[1:] - envelope_indices[:-1]) != 1)[0][0]

    v2 = v[envelope_indices[index_v1+1]]
    v1 = v[index_v1]

    pr = lambda rho, T: rho**2 * dAdrho_scalar(rho, T) 
    p_VLE = pr((1/v2 * rho_c), (T * T_c)) / p_c 

    return p_VLE, v1, v2

def compute_VLE():
    """
        compute the VLE region to an existing p-v plot
    """
    T_plot = jnp.linspace(0.6, 0.99, 20)
    T_plot = jnp.flip(T_plot)

    p_VLE_arr   = [1.0]
    v1_arr      = [1.0]
    v2_arr      = [1.0]

    for T in T_plot:
        p_VLE, v1, v2 = solve_VLE_pressure_in_interval(T, 0.01, 3.0)

        p_VLE_arr.append(p_VLE)
        v1_arr.append(v1)
        v2_arr.append(v2)

    return v1_arr, v2_arr, p_VLE_arr

if EOS != "IDEAL_GAS":
    v1_VLE, v2_VLE, p_VLE = compute_VLE()

def compute_isotherms_pv():
    num_points_plot = 1000

    T_normal = jnp.linspace(0.6, 1.7, 14)
    v_normal = jnp.logspace(jnp.log10(1/3+1e-2), jnp.log10(100), num_points_plot)
    p_vT = jax.vmap(jax.jit(lambda v, T: (1/v**2 * rho_c**2) * dAdrho_scalar(1/v * rho_c, T * T_c) /  p_c), (0,0), 0) #pressure function is vectorized for ndarrays

    isotherms = []

    for T in T_normal:
        p = p_vT(v_normal, T * jnp.ones(num_points_plot))
        isotherm = (T, v_normal, p)
        isotherms.append(isotherm)

    T = 1
    p = p_vT(v_normal, T * jnp.ones(num_points_plot))
    isotherms.append((T, v_normal, p))

    return isotherms

if EOS != "IDEAL_GAS":
    isotherms = compute_isotherms_pv()

def plot_pv(density, press, fig, ax):
    """
        Plot a p-v diagram for the reduced van der waals equations (nondimensionalized w.r.t. critical point)

        If points is a 2d-array it will be plotted as set of points in the p-v plane
    """
    ax.clear()

    #VLE region
    lcolor = 'tab:gray'
    lwidth = 1.5
    ax.plot(v1_VLE, p_VLE, '-', linewidth=lwidth, color=lcolor)
    ax.plot(v2_VLE, p_VLE, '-', linewidth=lwidth, color=lcolor)

    #critical isotherm
    isotherm_c = isotherms[-1]
    T, v, p = isotherm_c

    lcolor = 'tab:red'
    lwidth = 2.0
    ax.semilogx(v, p, linewidth=lwidth, color=lcolor)
    ax.semilogx(1,1,'o', markersize = 10, color=lcolor)

    #other isotherms
    lcolor = [0.5,0.5,0.5]
    lwidth = 0.5
    for isotherm in isotherms[:-1]:
        T, v, p = isotherm
        ax.semilogx(v, p, linewidth=lwidth, color=lcolor)

    #points
    density_normalized = density / rho_c
    press_normalized = press / p_c
    ax.scatter(1/density_normalized , press_normalized, marker = 'o', color = 'tab:orange')

    ax.set_xlim(1./3+1e-2,20)
    ax.set_ylim(0,4)
    ax.set_xlabel("reduced specific volume $v_r$")
    ax.set_ylabel("reduced pressure $p_r$")

    return ax
'''