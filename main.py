"""
    Main entry point for the application.
"""

if __name__ == "__main__":
    from prep_jax import *
    from modules.geometry.grid          import construct_mesh
    from modules.numerical.integration  import integrate, check_CFL
    from modules.simulation.initial     import initial_condition
    from modules.simulation.variables   import get_convert

    # prepare initial condition
    mesh                = construct_mesh()
    u, conversion       = initial_condition(mesh) 
    convert             = get_convert(conversion)
    u, T                = convert(u)
    del mesh

    from jax.numpy import sqrt

    rho = u[0]
    vel = sqrt((u[1] / rho)**2 + (u[2] / rho)**2)

    import matplotlib.pyplot as plt
    plt.imshow(vel.T, origin = 'lower')
    plt.show()

    '''
    u.block_until_ready()
    T.block_until_ready()
    print('finished initial condition')
    print('CFL: ', check_CFL(u, T).max())

    # simulate
    u, T = integrate(u, T) 
    '''
    
    
    
