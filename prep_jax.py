"""
    Prepare JAX for use in the current environment. 
    
    This should be used in all files where jax numpy is used.
"""

import jax
from functools import partial

from config.conf_jax import *

print('Workstation devices: ', jax.devices(backend="cpu"), jax.devices(backend="gpu"))

cpus = jax.devices("cpu")
gpus = jax.devices("gpu")

# Global flag to set a specific platform, must be used at startup.
#jax.config.update('jax_platform_name', 'cpu')

match DTYPE:
    case "DOUBLE":
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        DTYPE = jnp.float64
    case "SINGLE":
        import jax.numpy as jnp
        DTYPE = jnp.float32