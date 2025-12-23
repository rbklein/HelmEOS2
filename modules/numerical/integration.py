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

from modules.postprocess.post import init_postprocess, plot_postprocess, update_postprocess, NUM_ITS_PER_UPDATE
from modules.numerical.computation import midpoint_integrate

from functools import partial
from pathlib import Path

''' Consistency checks '''

KNOWN_TIME_STEP_METHODS = ["RK4", "FE", "WRAY", "BE"]

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
    case "WRAY":
        from modules.numerical.integrators.wray import Wray as time_step
    case "BE":
        from modules.numerical.integrators.BE import backward_euler as time_step
    case _:
        raise ValueError(f"Unknown time step method: {TIME_STEP_METHOD}")

def check_CFL(u, T):
    """
    Calculate the CFL number in every grid cell

    CFL is define as the max_i |u_i| * dt / dx
    """
    c = speed_of_sound(u[0], T)
    v_max = jnp.max(jnp.abs(u[1:(N_DIMENSIONS+1)] / u[0]), axis = 0) 
    cfl = (dt * (v_max + c)) / GRID_SPACING[0]
    return cfl

@jax.jit
def integrate(u, T):
    """
    Integrate the Compressible flow in time

    Parameters:
        - u (array-like): initial state
        - T (array-like): initial temperature associated to initial state

    Returns:
        Final state and temperature
    """
    # Define process status function
    status = lambda it, u, T: jax.debug.print(
        "Current time step: {it}/{its}, t: {t}, CFL: {cfl}", 
        it=it, 
        its = NUM_TIME_STEPS, 
        t=(it*dt), 
        cfl = jnp.max(check_CFL(u, T))
    )

    def step(_, carry):
        t, it, u_prev, T_prev = carry           # Unpack the carry variable
        u = time_step(u_prev, T_prev, dt, t)    # Compute new state
        T = temperature(u, T_prev)              # Compute new temperature using previous temperature as initial guess
        it = it + 1
        t  = t + dt
        jax.lax.cond((it % NUM_ITS_PER_UPDATE) == 0, 
                    lambda _: status(it, u, T), 
                    lambda _: None, 
                    operand=None)
        return (t, it, u, T)

    status(0, u, T)

    # Perform the integration over the specified number of time steps
    t, it, u, T = jax.lax.fori_loop(
        0, NUM_TIME_STEPS, step, (0.0, 0, u, T) #, None, length=NUM_TIME_STEPS
    )  

    return u, T


@jax.jit
def integrate_data(u, T):
    """
    Integrate the Compressible flow in time while gather all data

    Parameters:
        - u (array-like): initial state
        - T (array-like): initial temperature associated to initial state

    Returns:
        Final state and temperature
    """
    # Define process status function
    status = lambda it, u, T: jax.debug.print(
        "Current time step: {it}/{its}, t: {t}, CFL: {cfl}", 
        it=it, 
        its = NUM_TIME_STEPS, 
        t=(it*dt), 
        cfl = jnp.max(check_CFL(u, T))
    )

    def scan_step(carry, _):
        t, it, u_prev, T_prev = carry           # Unpack the carry variable
        u = time_step(u_prev, T_prev, dt, t)    # Compute new state
        T = temperature(u, T_prev)              # Compute new temperature using previous temperature as initial guess
        it = it + 1
        t  = t + dt
        jax.lax.cond((it % NUM_ITS_PER_UPDATE) == 0, 
                    lambda _: status(it, u, T), 
                    lambda _: None, 
                    operand=None)
        return (t, it, u, T), (u, T)

    u0, T0 = jnp.copy(u), jnp.copy(T)
    status(0, u, T)

    # Perform the integration over the specified number of time steps
    (t, it, u, T), (u_hist, T_hist) = jax.lax.scan(
        scan_step, (0.0, 0, u, T), None, length=NUM_TIME_STEPS
    )  

    u_hist = jnp.concatenate([u0[jnp.newaxis, ...], u_hist], axis=0)
    T_hist = jnp.concatenate([T0[jnp.newaxis, ...], T_hist], axis=0)

    return u, T, u_hist, T_hist



def integrate_interactive(u, T):
    """
    Interactive integration using a JIT-compiled jax.lax.scan for each interval
    between update_postprocess calls.
    """

    @jax.jit
    def step(u, T, dt, t):
        u = time_step(u, T, dt, t)
        T = temperature(u, T)
        t = t + dt
        return u, T, t

    status = lambda it, u, T: print(f"Current time step: {it}/{NUM_TIME_STEPS}, t: {it*dt}, CFL: {jnp.max(check_CFL(u, T)):.4f}")

    scan_steps = NUM_ITS_PER_UPDATE
    total_steps = NUM_TIME_STEPS

    fig, plot_grid = init_postprocess()
    plot_grid = plot_postprocess(u, T, fig, plot_grid, cmap=COLORMAP)
    status(0, u, T)

    #compiled interval of steps between plot updates
    @partial(jax.jit, static_argnames=['steps'])
    def compiled_step(u, T, steps, t):
        u, T, t = jax.lax.scan(lambda carry, _: (step(carry[0], carry[1], dt, carry[2]), None), (u, T, t), None, length=steps)[0]
        return u, T, t
    
    t = 0.0
    steps_done = 0
    while steps_done < total_steps:
        #steps = min(scan_steps, total_steps - steps_done)
        u, T, t = compiled_step(u, T, scan_steps, t)
        steps_done += scan_steps

        status(steps_done, u, T)
        update_postprocess(u, T, fig, plot_grid)

    return u, T