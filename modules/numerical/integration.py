"""
    Functions for numerical integration.
"""

from prep_jax import *
from config.conf_numerical import *
from config.conf_simulation import *
from config.conf_postprocess import *

from config.conf_geometry import *
from config.conf_thermodynamics import *

from modules.thermodynamics.EOS import *
from modules.geometry.grid import GRID_SPACING

''' Consistency checks '''

KNOWN_TIME_STEP_METHODS = ["RK4", "FE"]

assert TIME_STEP_METHOD in KNOWN_TIME_STEP_METHODS, f"Unknown time step method: {TIME_STEP_METHOD}"
assert isinstance(NUM_TIME_STEPS, int) and NUM_TIME_STEPS > 0, "NUM_TIME_STEPS must be a positive integer"
assert isinstance(TOTAL_TIME, (int, float)) and TOTAL_TIME > 0, "TOTAL_TIME must be a positive number"

''' Derived parameters '''

dt = TOTAL_TIME / NUM_TIME_STEPS  # Time step size

''' Functions for numerical integration '''

match TIME_STEP_METHOD:
    case "RK4":
        from modules.numerical.integrators.RK4 import RK4 as time_step
    case "FE":
        from modules.numerical.integrators.FE import forward_euler as time_step
    case _:
        raise ValueError(f"Unknown time step method: {TIME_STEP_METHOD}")

def check_CFL(u):
    """
    Calculate the CFL number in every grid cell

    CFL is define as the max_i |u_i| * dt / dx
    """
    T = temperature(u)
    c = speed_of_sound(u[0], T)
    v_max = jnp.max(jnp.abs(u[1:(N_DIMENSIONS+1)] / u[0]), axis = 0) 
    cfl = (dt * (v_max + c)) / GRID_SPACING[0]
    return cfl

from modules.postprocess.post import init_postprocess, plot_postprocess, update_postprocess, NUM_ITS_PER_UPDATE
from modules.numerical.computation import midpoint_integrate

@jax.jit
def integrate(u):

    def scan_step(carry, _):
        it, u = carry  # Unpack the carry variable
        jax.lax.cond((it % NUM_ITS_PER_UPDATE) == 0, 
                    lambda _: jax.debug.print("Current time step: {it}/{its}, t: {t}, CFL: {cfl}", it=it, its = NUM_TIME_STEPS, t=(it*dt), cfl = jnp.max(check_CFL(u))), 
                    lambda _: None, 
                    operand=None)
        return (it+1, time_step(u, dt)), None

    it, u = jax.lax.scan(
        scan_step, (0, u), None, length=NUM_TIME_STEPS
    )[0]  # Perform the integration over the specified number of time steps
    return u

def integrate_interactive(u):
    """
    Interactive version of the integrate function that plots postprocessing information 

    Use comes at the cost of possible optimizations like loop unrolling
    """ 

    step = jax.jit(time_step)

    fig, plot_grid = init_postprocess()
    plot_grid = plot_postprocess(u, fig, plot_grid, cmap = COLORMAP)
    print(f"Current time step: {0}/{NUM_TIME_STEPS}, t: {0}, CFL: {jnp.max(check_CFL(u)):.4f}")

    #use a regular loop to integrate the system
    for it in range(NUM_TIME_STEPS):
        if (it % NUM_ITS_PER_UPDATE) == 0 and it != 0:
            print(f"Current time step: {it}/{NUM_TIME_STEPS}, t: {it*dt}, CFL: {jnp.max(check_CFL(u)):.4f}")
            update_postprocess(u, fig, plot_grid)

        u = step(u, dt)

    return u

from pathlib import Path

def integrate_experiment(u):
    """
    A time-integrator that can be customized to perform numerical experiments
    """

    #step = jax.jit(time_step)

    u0 = jnp.copy(u)

    rho = u[0]
    T   = temperature(u)
    S0  = midpoint_integrate(rho * entropy(rho, T))
    p_exact = 2 * p_c

    arr_e_s = jnp.zeros(NUM_TIME_STEPS + 1)
    arr_e_s = arr_e_s.at[0].set(0.0)

    arr_e_p = jnp.zeros(NUM_TIME_STEPS + 1)
    arr_e_p = arr_e_p.at[0].set(0.0)


    def scan_step(carry, _):
        it, u, arr_e_s, arr_e_p = carry

        rho = u[0]
        T   = temperature(u)
        S   = midpoint_integrate(rho * entropy(rho, T))

        p   = pressure(rho, T)

        arr_e_s = arr_e_s.at[it].set(jnp.abs(S - S0) / jnp.abs(S0))
        arr_e_p = arr_e_p.at[it].set(jnp.sqrt(midpoint_integrate((p - p_exact)**2)) / (jnp.abs(p_exact) * 1))

        jax.debug.print("Current time step: {it}/{its}, t: {t}", it=it, its = NUM_TIME_STEPS, t=(it*dt))

        return (it+1, time_step(u, dt), arr_e_s, arr_e_p), None

    it, u, arr_e_s, arr_e_p = jax.lax.scan(
        scan_step, (0, u, arr_e_s, arr_e_p), None, length=NUM_TIME_STEPS
    )[0]

    err = u - u0
    name = (
        f"{NUMERICAL_FLUX}_{DISCRETE_GRADIENT}_{NAME_MOLECULE}_"
        f"{EOS}_{GRID_RESOLUTION[0]}_{NUM_TIME_STEPS}"
    )

    base = Path("./output") / TEST_CASE
    path_sol  = base / "solution_data"
    path_err  = base / "convergence_data"
    path_ent  = base / "entropy_data"
    path_pres = base / "pressure_data"

    # Ensure each directory exists
    for p in (path_sol, path_err, path_ent, path_pres):
        p.mkdir(parents=True, exist_ok=True)

    # Save files (Path objects are fine)
    jnp.save(path_sol  / f"{name}.npy", u)
    jnp.save(path_err  / f"{name}.npy", err)
    jnp.save(path_ent  / f"{name}.npy", arr_e_s)
    jnp.save(path_pres / f"{name}.npy", arr_e_p)
    return u