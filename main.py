"""
    Main entry point for the application.
"""

ns = 10
from time import sleep
def sleep_for(ns):
    for i in range(ns):
        sleep(1)
        print(i)

if __name__ == "__main__":
    from prep_jax import *
    from modules.geometry.grid          import construct_mesh
    from modules.numerical.integration  import integrate
    from modules.simulation.initial     import initial_condition
    from modules.simulation.variables   import get_convert
    from modules.thermodynamics.EOS     import molecule
    #from modules.postprocess.post       import init_postprocess, plot_postprocess, COLORMAP, show

    #from jax import profiler
    #profiler_dir = "./jax_profile_logs/"
    #profiler.start_trace(profiler_dir)

    # prepare initial condition
    mesh                = construct_mesh()
    u, conversion       = initial_condition(mesh) 
    convert             = get_convert(conversion)
    u, T                = convert(u)

    u.block_until_ready()
    print('finished initial condition')

    #sleep_for(10)

    rho_c, T_c, p_c = molecule.critical_point
    print('rho_c: ', rho_c)
    print('T_c: ', T_c)
    print('p_c: ', p_c)

    # simulate
    u, T = integrate(u, T) 

    u.block_until_ready()
    T.block_until_ready()

    import matplotlib.pyplot as plt
    from jax.numpy.linalg import norm
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(u[0].T / rho_c, origin = 'lower', cmap = 'magma')
    ax[0,1].imshow(norm(u[1:3], axis = 0).T, origin = 'lower', cmap = 'magma')
    ax[1,0].imshow(u[3].T, origin = 'lower', cmap = 'magma')
    ax[1,1].imshow(T.T / T_c, origin = 'lower', cmap = 'magma')
    plt.show()

    #profiler.stop_trace()

    #from jax.numpy import save
    #save("test_u.npy", u)
    #save("test_T.npy", T)

    # postprocess
    #fig, plot_grid  = init_postprocess()
    #plot_grid       = plot_postprocess(u, T, fig, plot_grid, cmap=COLORMAP, freeze_image=True)
    #show()
    
    