"""
    Main entry point for the application.
"""

if __name__ == "__main__":
    from prep_jax import *
    from modules.geometry.grid          import mesh
    from modules.numerical.integration  import integrate
    from modules.simulation.initial     import initial_condition
    from modules.simulation.variables   import convert
    from modules.thermodynamics.EOS     import molecule
    from modules.postprocess.post       import init_postprocess, plot_postprocess, COLORMAP, show

    # prepare initial condition
    initial, conversion = initial_condition(mesh) 
    u, T                = convert(initial, conversion)

    u.block_until_ready()
    print('finished initial condition')

    if len(mesh) > 1:
        del mesh

    rho_c, T_c, p_c = molecule.critical_point
    print('rho_c: ', rho_c)
    print('T_c: ', T_c)
    print('p_c: ', p_c)

    # simulate
    u, T = integrate(u, T) 

    u.block_until_ready()
    T.block_until_ready()
    print('finished timestepping')
    
    jnp.save("test_u.npy", u)
    jnp.save("test_T.npy", T)

    # postprocess
    #fig, plot_grid  = init_postprocess()
    #plot_grid       = plot_postprocess(u, T, fig, plot_grid, cmap=COLORMAP, freeze_image=True)
    #show()
    