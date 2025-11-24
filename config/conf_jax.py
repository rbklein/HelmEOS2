"""
    Configure JAX settings
"""

DTYPE = "DOUBLE"

SHARD_ARRAYS    = True
SHARD_DEVICES   = 'GPU'
SHARD_TOPOLOGY  = (2, 2)
SHARD_LABELS    = ('a','b')
SHARD_PARTITION = ('a','b', None) # can be automated but lots of work