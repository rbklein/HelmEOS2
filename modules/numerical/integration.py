"""
    Functions for numerical integration.
"""

from prep_jax                   import *
from config.conf_numerical      import *
from config.conf_simulation     import *
from config.conf_postprocess    import *
from config.conf_geometry       import *
from config.conf_thermodynamics import *

from modules.thermodynamics.EOS     import temperature, speed_of_sound
from modules.geometry.grid          import GRID_SPACING
from jax.numpy                      import max, abs
from jax                            import jit
from jax.lax                        import fori_loop, cond
from jax.debug                      import print as jax_print

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
        #from modules.numerical.integrators.BE import backward_euler as time_step
        raise NotImplementedError(f"Backward Euler out of order")
    case _:
        raise ValueError(f"Unknown time step method: {TIME_STEP_METHOD}")

def check_CFL(u, T):
    """
    Calculate the CFL number in every grid cell

    CFL is define as the max_i |u_i| * dt / dx
    """
    c = speed_of_sound(u[0], T)
    v_max = max(abs(u[1:(N_DIMENSIONS+1)] / u[0]), axis = 0) 
    cfl = (dt * (v_max + c)) / GRID_SPACING[0]
    return cfl

@jit
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
    status = lambda it, u, T: jax_print(
        "Current time step: {it}/{its}, t: {t}, CFL: {cfl}", 
        it=it, 
        its = NUM_TIME_STEPS, 
        t=(it*dt), 
        cfl = max(check_CFL(u, T))
    )
    
    def step(_, carry):
        t, it, u_prev, T_prev = carry           # Unpack the carry variable
        u = time_step(u_prev, T_prev, dt, t)    # Compute new state
        T = temperature(u, T_prev)              # Compute new temperature using previous temperature as initial guess
        it = it + 1
        t  = t + dt
        cond((it % NUM_ITS_PER_UPDATE) == 0, 
                    lambda _: status(it, u, T), 
                    lambda _: None, 
                    operand=None)
        return (t, it, u, T)

    status(0, u, T)

    # carry = (0.0, 0, u, T)
    # for _ in range(NUM_TIME_STEPS):
    #     carry = step(carry)
    #     if (carry[1] % NUM_ITS_PER_UPDATE) == 0:
    #         status(carry[1], carry[2], carry[3])
    
    # t, it, u, T = carry

    # Perform the integration over the specified number of time steps
    t, it, u, T = fori_loop(
        0, NUM_TIME_STEPS, step, (0.0, 0, u, T)
    )  

    return u, T