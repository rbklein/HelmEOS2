"""
    Configure JAX settings
"""

# Options: "SINGLE", "DOUBLE"
USE_DTYPE = "DOUBLE" 

SHARD_ARRAYS    = False
SHARD_DEVICES   = 'GPU'
SHARD_TOPOLOGY  = (2, 2)
SHARD_LABELS    = ('a', 'b')
SHARD_PARTITION = ('a', 'b', None) # can be automated but lots of work