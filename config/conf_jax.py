"""
    Configure JAX settings
"""

DTYPE = "DOUBLE"

SHARD_ARRAYS    = True
SHARD_DEVICES   = 'GPU'
SHARD_TOPOLOGY  = (1,)
SHARD_LABELS    = ('a',)
SHARD_PARTITION = ('a', None, None) # can be automated but lots of work