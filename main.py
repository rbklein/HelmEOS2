"""
    Main entry point for the application.
"""

if __name__ == "__main__":
    from prep_jax import *
    from modules.geometry.grid          import construct_mesh
    from modules.numerical.integration  import integrate_data, check_CFL
    from modules.simulation.initial     import initial_condition
    from modules.simulation.variables   import get_convert

    # prepare initial condition
    mesh                = construct_mesh()
    u, conversion       = initial_condition(mesh) 
    convert             = get_convert(conversion)
    u, T                = convert(u)
    del mesh

    u.block_until_ready()
    T.block_until_ready()
    print('Finished initial condition')
    print('CFL: ', check_CFL(u, T).max())

    # simulate
    u, T, data = integrate_data(u, T) 

    import matplotlib.pyplot    as plt
    import jax.numpy            as jnp

    from config.conf_thermodynamics import molecule
    rho_c, T_c, p_c = molecule.critical_point

    from modules.thermodynamics.EOS import c_p, pressure

    fig, ax = plt.subplots(2, 2)

    im0 = ax[0, 0].plot(u[0, :] / rho_c)
    im1 = ax[0, 1].plot(T / T_c)

    #vel = jnp.sqrt((u[1, :, :] / u[0, :, :])**2 + (u[2, :, :] / u[0, :, :])**2)
    vel = u[1] / u[0]

    im2 = ax[1, 0].plot(pressure(u[0, :], T) / p_c)
    im2 = ax[1, 0].plot(vel / 10)

    im3 = ax[1, 1].plot(c_p(u[0, :], T))

    fig.tight_layout()

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(data[:,0], data[:,1])
    ax[1].plot(data[:,0], data[:,2])

    plt.show()

    
    

    