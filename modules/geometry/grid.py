"""
    Functions for the grid geometry.
"""

from prep_jax               import *
from config.conf_geometry   import *

''' Consistency checks '''

N_DIMENSIONS = int(N_DIMENSIONS)  # Ensure n_dimensions is an integer
assert N_DIMENSIONS >= 1, "Number of dimensions must be a positive integer"
assert len(DOMAIN_SIZE) == N_DIMENSIONS, "Domain size must match the number of dimensions"
assert len(GRID_RESOLUTION) == N_DIMENSIONS, "Grid resolution must match the number of dimensions"
assert all(isinstance(v, int) for v in GRID_RESOLUTION), "Grid resolution must be integers"


''' Derived parameters '''
from jax        import device_put
from jax.numpy  import prod, array, linspace, meshgrid
from numpy      import meshgrid as np_meshgrid


GRID_SPACING = tuple(size / res for size, res in zip(DOMAIN_SIZE, GRID_RESOLUTION))  # Spacing between grid points in each dimension
CELL_VOLUME  = prod(array(GRID_SPACING))  # Volume of each grid cell

''' Functions '''

def construct_mesh():
    '''
    Construct the mesh in a single- or multi-device setting
    '''
    if SHARD_ARRAYS:
        assert len(SHARD_PARTITION) == len(GRID_RESOLUTION), "Sharding memory partition is not compatible with grid"

        # Construct mesh on cpu and pass around sharded
        mesh = np_meshgrid(
            *[linspace(spacing/2, size - spacing/2, num) for size, num, spacing in zip(DOMAIN_SIZE, GRID_RESOLUTION, GRID_SPACING)], 
            indexing='ij'
        )

        mesh = [device_put(mesh[i], NamedSharding(device_mesh, PartitionSpec(*SHARD_PARTITION))) for i in range(N_DIMENSIONS)]
    else:
        mesh = meshgrid(
            *[linspace(spacing/2, size - spacing/2, num) for size, num, spacing in zip(DOMAIN_SIZE, GRID_RESOLUTION, GRID_SPACING)], 
            indexing='ij'
        )  #meshgrid for the grid points

    return mesh






