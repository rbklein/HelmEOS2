"""
    Configure JAX settings
"""

DTYPE = "DOUBLE"

SHARD_ARRAYS    = False
SHARD_DEVICES   = 'GPU'
SHARD_TOPOLOGY  = (1, 4)
SHARD_LABELS    = ('a','b')
SHARD_PARTITION = ('a','b', None) # can be automated but lots of work