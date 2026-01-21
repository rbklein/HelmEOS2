"""
    Prepare JAX for use in the current environment. 
    
    This should be used in all files where jax numpy is used.
"""

from config.conf_jax import *


import os

# Global flag to turn off memory preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

# Global flag to use CPU
os.environ['JAX_PLATFORMS'] = 'cpu'

from jax import devices
#print('Workstation devices: ', devices(backend="cpu"), devices(backend="gpu")) #should by try catch type formulation

cpus = devices("cpu")
gpus = [] # devices("gpu") # 


match USE_DTYPE:
    case "DOUBLE":
        from jax.numpy import float64 as DTYPE
        from jax import config
        config.update("jax_enable_x64", True)
    case "SINGLE":
        from jax.numpy import float32 as DTYPE

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
