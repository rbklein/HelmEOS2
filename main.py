"""
    Main entry point for the application.
"""

if __name__ == "__main__":
    from prep_jax import *
    from modules.geometry.grid          import mesh
    from modules.numerical.integration  import integrate, integrate_interactive, integrate_experiment
    from modules.simulation.initial     import initial_condition
    from modules.thermodynamics.EOS     import temperature_eos, total_energy, rho_c, p_c, T_c, speed_of_sound

    #   rho, velocity, pressure 
    #   clean this sequence up
    initial = initial_condition(mesh, rho_c, p_c)  
    p   = initial[3]
    rho = initial[0]
    T   = temperature_eos(rho, p)
    v   = initial[1:3, :, :]  
    m   = v * rho 
    E   = total_energy(rho, T, v)  
    u   = jnp.stack((rho, m[0], m[1], E), axis=0)  
    
    print(rho_c, p_c, T_c)

    from time import time
    start_time = time()
    u   = integrate_experiment(u) 
    end_time = time()

    



