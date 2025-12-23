"""
    Configure JAX settings
"""

# Options: "SINGLE", "DOUBLE"
DTYPE = "SINGLE" 

SHARD_ARRAYS    = False
SHARD_DEVICES   = 'GPU'
SHARD_TOPOLOGY  = (1,)
SHARD_LABELS    = ('a')
SHARD_PARTITION = ('a', None, None) # can be automated but lots of work