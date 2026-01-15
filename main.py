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
    print('finished initial condition')
    print('CFL: ', check_CFL(u, T).max())

    # simulate
    u, T, data = integrate_data(u, T) 
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(data[:, 0], label='Kinetic Energy')
    ax[1].plot(data[:, 1], label='Entropy')

    fig, ax = plt.subplots()
    ax.plot(u[0])
    plt.show()

    