"""
    Functions for the grid geometry.
"""

from config.conf_geometry import *
import numpy as np

''' Consistency checks '''

N_DIMENSIONS = int(N_DIMENSIONS)  # Ensure n_dimensions is an integer
assert N_DIMENSIONS >= 1, "Number of dimensions must be a positive integer"
assert len(DOMAIN_SIZE) == N_DIMENSIONS, "Domain size must match the number of dimensions"
assert len(GRID_RESOLUTION) == N_DIMENSIONS, "Grid resolution must match the number of dimensions"
assert all(isinstance(v, int) for v in GRID_RESOLUTION), "Grid resolution must be integers"


''' Derived parameters '''
from prep_jax import *

GRID_SPACING = tuple(size / res for size, res in zip(DOMAIN_SIZE, GRID_RESOLUTION))  # Spacing between grid points in each dimension
CELL_VOLUME  = jnp.prod(jnp.array(GRID_SPACING))  # Volume of each grid cell

''' Functions '''

def construct_mesh():
    '''
    Construct the mesh in a single- or multi-device setting
    '''
    if SHARD_ARRAYS:
        assert len(SHARD_PARTITION) == len(GRID_RESOLUTION), "Sharding memory partition is not compatible with grid"

        # Construct mesh on cpu and pass around sharded
        mesh = np.meshgrid(
            *[jnp.linspace(spacing/2, size - spacing/2, num) for size, num, spacing in zip(DOMAIN_SIZE, GRID_RESOLUTION, GRID_SPACING)], 
            indexing='ij'
        )

        mesh = [jax.device_put(mesh[i], NamedSharding(device_mesh, PartitionSpec(*SHARD_PARTITION))) for i in range(N_DIMENSIONS)]
    else:
        mesh = jnp.meshgrid(
            *[jnp.linspace(spacing/2, size - spacing/2, num) for size, num, spacing in zip(DOMAIN_SIZE, GRID_RESOLUTION, GRID_SPACING)], 
            indexing='ij'
        )  #meshgrid for the grid points

    return mesh






