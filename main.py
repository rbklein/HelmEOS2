"""
    Main entry point for the application.
"""

if __name__ == "__main__":
    from prep_jax import *
    from modules.geometry.grid          import mesh
    from modules.numerical.integration  import integrate, integrate_interactive, integrate_experiment
    from modules.simulation.initial     import initial_condition
    from modules.simulation.variables   import convert
    from modules.thermodynamics.EOS     import molecule
    from modules.postprocess.post       import init_postprocess, plot_postprocess, COLORMAP, show

    # prepare initial condition (density, velocity, pressure)
    initial = initial_condition(mesh, molecule)  
    u, T    = convert(initial, 'rvp')

    rho_c, T_c, p_c = molecule.critical_points
    print(r'$\rho_c$: ', rho_c)
    print(r'$T_c$: ', T_c)
    print(r'$p_c$: ', p_c)

    # simulate
    #u, T = integrate_interactive(u, T) 

    #negative values due to reference choice?

    # postprocess
    fig, plot_grid  = init_postprocess()
    plot_grid       = plot_postprocess(u, T, fig, plot_grid, cmap=COLORMAP, freeze_image=True)
    show()




