"""
    Prepare JAX for use in the current environment. 
    
    This should be used in all files where jax numpy is used.
"""

# Global flag to turn off memory preallocation
# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from functools import partial

from config.conf_jax import *

print('Workstation devices: ', jax.devices(backend="cpu"), jax.devices(backend="gpu"))

cpus = jax.devices("cpu")
gpus = jax.devices("gpu")

# Global flag to set a specific platform, must be used at startup.
# jax.config.update('jax_platform_name', 'cpu')

match DTYPE:
    case "DOUBLE":
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        DTYPE = jnp.float64
    case "SINGLE":
        import jax.numpy as jnp
        DTYPE = jnp.float32

if SHARD_ARRAYS:
    from jax.experimental import mesh_utils
    from jax.sharding     import Mesh, NamedSharding, PartitionSpec
    
    from math import prod

    assert len(SHARD_TOPOLOGY) == len(SHARD_LABELS), "Incompatible device topology and axis labels"

    # Verify correct device for sharding
    supported_devices = ['GPU', 'CPU']
    assert SHARD_DEVICES in supported_devices, "Unknown or unsupported device type {SHARD_DEVICES}"

    # Set device array
    match SHARD_DEVICES:
        case 'GPU':
            devices = gpus
        case 'CPU':
            devices = cpus

    # Check if device topology is compatible
    num_devices = prod(SHARD_TOPOLOGY)
    assert num_devices >= len(devices), "Device topology for sharding incompatible with machine"

    device_mesh = mesh_utils.create_device_mesh(SHARD_TOPOLOGY, devices = devices[:num_devices])
    device_mesh = Mesh(device_mesh, SHARD_LABELS)
