"""
    Main entry point for the application.
"""

if __name__ == "__main__":
    from prep_jax import *
    from modules.geometry.grid          import mesh
    from modules.numerical.integration  import integrate, integrate_interactive, integrate_experiment
    from modules.simulation.initial     import initial_condition
    from modules.thermodynamics.EOS     import rho_c, p_c, T_c, temperature_rpt, total_energy
    from modules.postprocess.post       import init_postprocess, plot_postprocess, COLORMAP, show

    from modules.thermodynamics.gas_models.ideal_gas import temperature_rpt_ideal

    #   rho, velocity, pressure 
    #   clean this sequence up
    initial = initial_condition(mesh, rho_c, p_c)  
    p   = initial[3]
    rho = initial[0]

    Tguess = temperature_rpt_ideal(rho, p, None)
    T   = temperature_rpt(rho, p, Tguess)

    v   = initial[1:3, :, :]  
    m   = v * rho 
    E   = total_energy(rho, T, v)  
    u   = jnp.stack((rho, m[0], m[1], E), axis=0)   

    u, T = integrate(u, T) 

    fig, plot_grid = init_postprocess()
    plot_grid = plot_postprocess(u, T, fig, plot_grid, cmap=COLORMAP, freeze_image=True)
    show()

