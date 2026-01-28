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
from jax.numpy                      import max, abs, stack
from jax                            import jit
from jax.lax                        import fori_loop, scan, cond
from jax.debug                      import print as jax_print

from modules.postprocess.derived_quantities.vorticity import total_enstrophy, total_entropy, total_kinetic_energy, pressure_work, total_dilation
from modules.simulation.boundary import apply_temperature_boundary_condition

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

    # Perform the integration over the specified number of time steps
    t, it, u, T = fori_loop(
        0, NUM_TIME_STEPS, step, (0.0, 0, u, T)
    )  

    status(it, u, T)

    return u, T


@jit
def integrate_data(u, T):
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

    def step(carry, _):
        t, it, u_prev, T_prev = carry           # Unpack the carry variable
        u = time_step(u_prev, T_prev, dt, t)    # Compute new state
        T = temperature(u, T_prev)              # Compute new temperature using previous temperature as initial guess
        t  = t + dt
        it = it + 1

        k = total_kinetic_energy(u, T)
        s = total_entropy(u, T)
        #p = total_enstrophy(u, T)

        Tp = apply_temperature_boundary_condition(u, T)
        n_x, n_y, n_z = GRID_RESOLUTION

        dxT = abs(Tp[1:, 1:-1, 1:-1] - Tp[:-1, 1:-1, 1:-1]) < 1e-3
        dyT = abs(Tp[1:-1, 1:, 1:-1] - Tp[1:-1, :-1, 1:-1]) < 1e-3
        dzT = abs(Tp[1:-1, 1:-1, 1:] - Tp[1:-1, 1:-1, :-1]) < 1e-3
        dxT = dxT[1:, :, :] + dxT[:-1, :, :]
        dyT = dyT[:, 1:, :] + dyT[:, :-1, :]
        dzT = dzT[:, :, 1:] + dzT[:, :, :-1]

        dT  = (dxT + dyT + dzT) >= 1
        percentage = sum(dT) / prod(GRID_RESOLUTION)

        data = stack((t, k, s, percentage))#, p))

        cond((it % NUM_ITS_PER_UPDATE) == 0, 
                    lambda _: status(it, u, T), 
                    lambda _: None, 
                    operand=None)
        
        return (t, it, u, T), data


    # Perform the integration over the specified number of time steps
    (t, its, u, T), data = scan(
        step, (0.0, 0, u, T), None, length = NUM_TIME_STEPS
    )  

    return u, T, data



@jit(static_argnames=['num_steps_per_conv', 'num_convs'])
def integrate_5convs(u, T, it, t, num_steps_per_conv, num_convs):
    """
    Integrate 5 convective time scales of the taylor green vortex
    """
    # Define process status function
    status = lambda it, u, T: jax_print(
        "Current time step: {it}/{its}, t: {t}, CFL: {cfl}", 
        it=it, 
        its = NUM_TIME_STEPS, 
        t=(it*dt), 
        cfl = max(check_CFL(u, T))
    )

    def step(carry, _):
        t, it, u_prev, T_prev = carry           # Unpack the carry variable
        u = time_step(u_prev, T_prev, dt, t)    # Compute new state
        T = temperature(u, T_prev)              # Compute new temperature using previous temperature as initial guess
        t  = t + dt
        it = it + 1

        k = total_kinetic_energy(u, T)
        s = total_entropy(u, T)
        En = total_enstrophy(u, T)
        di = total_dilation(u, T)
        pw = pressure_work(u, T)

        data = stack((k, s, En, di, pw))

        cond((it % NUM_ITS_PER_UPDATE) == 0, 
                    lambda _: status(it, u, T), 
                    lambda _: None, 
                    operand=None)
        
        return (t, it, u, T), data
    
    # Perform the integration over the specified number of time steps
    (t, its, u, T), data = scan(
        step, (t, it, u, T), None, length = num_convs * num_steps_per_conv
    )  

    return u, T, data


def integrate_TG(u, T):
    """
    Integrate the Compressible flow in time

    Parameters:
        - u (array-like): initial state
        - T (array-like): initial temperature associated to initial state

    Returns:
        Final state and temperature
    """    
    from config.conf_simulation import _num_conv_times, _conv_time
    from config.conf_numerical  import _num_conv_times as _nct_num, _num_steps_per_conv

    from jax.numpy              import save

    assert _num_conv_times == _nct_num, "Number of convective time scales is not the same in configuration files"

    for i in range(4):
        if i == 1:
            u, T, data = integrate_5convs(u, T, 5 * i * _num_steps_per_conv, 5 * i * _conv_time, _num_steps_per_conv, 3)
            print('saving intermediate...')
            file_name_u     = "sim_data/u_01_1600_2_5.npy"
            file_name_T     = "sim_data/T_01_1600_2_5.npy"
            file_name_data  = "sim_data/data_01_1600_2_5.npy"
            save(file_name_u, u)
            save(file_name_T, T)
            save(file_name_data, data)
            u, T, data = integrate_5convs(u, T, (5 * i + 3) * _num_steps_per_conv, (5 * i + 3) * _conv_time, _num_steps_per_conv, 2)

        else:
            u, T, data = integrate_5convs(u, T, 5 * i * _num_steps_per_conv, 5 * i * _conv_time, _num_steps_per_conv, 5)

        print('saving...')
        file_name_u     = "sim_data/u_01_1600_" + str(i+1) + ".npy"
        file_name_T     = "sim_data/T_01_1600_" + str(i+1) + ".npy"
        file_name_data  = "sim_data/data_01_1600_" + str(i+1) + ".npy"

        save(file_name_u, u)
        save(file_name_T, T)
        save(file_name_data, data)


    return u, T, data



# # cond((it % NUM_ITS_PER_UPDATE) == 0, 
#         #             lambda _: status(it, u, T), 
#         #             lambda _: None, 
#         #             operand=None)
