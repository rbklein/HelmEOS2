"""
    Main entry point for the application.
"""

if __name__ == "__main__":
    from prep_jax import *
    from modules.geometry.grid          import construct_mesh
    from modules.numerical.integration  import integrate, check_CFL
    from modules.simulation.initial     import initial_condition
    from modules.simulation.variables   import get_convert
    from modules.thermodynamics.EOS     import molecule
    #from modules.postprocess.post       import init_postprocess, plot_postprocess, COLORMAP, show

    # prepare initial condition
    mesh                = construct_mesh()
    u, conversion       = initial_condition(mesh) 
    convert             = get_convert(conversion)
    u, T                = convert(u)
    del mesh

    u.block_until_ready()
    print('finished initial condition')
    print('CFL: ', check_CFL(u, T).max())

    # simulate
    u, T, data = integrate(u, T) 

    u.block_until_ready()
    T.block_until_ready()
    data.block_until_ready()

    from jax.numpy import save
    save("test_u.npy", u)
    save("test_T.npy", T)
    # save("data.npy", data)

    # postprocess
    # fig, plot_grid  = init_postprocess()
    # plot_grid       = plot_postprocess(u, T, fig, plot_grid, cmap=COLORMAP, freeze_image=True)
    # show()
    
    