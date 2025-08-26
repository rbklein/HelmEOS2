"""
    Functions for numerical integration.
"""

from prep_jax import *
from config.conf_numerical import *
from config.conf_simulation import *
from config.conf_postprocess import *

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


#def integrate_debug(u):
    """
    Debug version of the integrate function that prints debug information.
    """ 

    import matplotlib.pyplot as plt
    import numpy as np

    #please clean up
    from modules.postprocess.derived_quantities.vorticity import vorticity_2d as vorticity

    step = jax.jit(time_step)
    
    plt.ion()
    fig, axes = plt.subplots(1,4, figsize=(20, 5))
    im1 = axes[0].imshow(u[0, :, :].T, origin = 'lower', cmap='plasma')
    cbar1 = fig.colorbar(im1, ax=axes[0], label='Density')

    im2 = axes[1].imshow(temperature(u).T, origin = 'lower', cmap='plasma')
    cbar2 = fig.colorbar(im2, ax=axes[1], label='Temperature')

    im3 = axes[2].imshow(jnp.linalg.norm(u[1:3, :, :], axis=0).T, origin = 'lower', cmap='plasma')
    cbar3 = fig.colorbar(im3, ax=axes[2], label='Velocity Magnitude')

    im4 = axes[3].imshow(vorticity(u).T, origin = 'lower', cmap='plasma')
    cbar4 = fig.colorbar(im4, ax=axes[3], label='Vorticity')

    #use a regular loop to integrate the system
    for it in range(NUM_TIME_STEPS):
        if (it % 10) == 0:
            print(f"Current time step: {it}/{NUM_TIME_STEPS}, t: {it*dt}, CFL: {jnp.max(check_CFL(u)):.4f}")

            # Updated data
            data1 = np.array(u[0, :, :]).T
            data2 = np.array(temperature(u)).T
            data3 = np.array(jnp.linalg.norm(u[1:3, :, :], axis=0)).T
            data4 = np.array(vorticity(u)).T

            # Update image data
            im1.set_data(data1)
            im2.set_data(data2)
            im3.set_data(data3)
            im4.set_data(data4)

            # Update color limits to fit new data
            im1.set_clim(vmin=data1.min(), vmax=data1.max())
            im2.set_clim(vmin=data2.min(), vmax=data2.max())
            im3.set_clim(vmin=data3.min(), vmax=data3.max())
            im4.set_clim(vmin=data4.min(), vmax=data4.max())

            # Update colorbars
            cbar1.update_normal(im1)
            cbar2.update_normal(im2)
            cbar3.update_normal(im3)
            cbar4.update_normal(im4)

            fig.canvas.draw_idle()
            plt.pause(0.001)  # Pause to update the plot

        u = step(u, dt)

    return u
