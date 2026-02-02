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
    u0, T0                = convert(u)
    del mesh

    u0.block_until_ready()
    T0.block_until_ready()
    print('Finished initial condition')
    print('CFL: ', check_CFL(u0, T0).max())

    # simulate
    u, T, data = integrate_data(u0, T0) 

    import numpy as np

    np.save('experiment_data/density_wave/data_VdW_32.npy', data)
    np.save('experiment_data/density_wave/u_VdW_32.npy', u)
    np.save('experiment_data/density_wave/T_VdW_32.npy', T)

    