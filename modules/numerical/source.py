"""
    Functions for calculating source terms in numerical simulations.
"""

from prep_jax               import *
from config.conf_numerical  import *
from config.conf_geometry   import *

from jax import jit

''' Consistency checks '''

KNOWN_SOURCE_TERMS = [
    "NONE",
    "MMS1",
]

assert SOURCE_TERM in KNOWN_SOURCE_TERMS, f"Unknown source term: {SOURCE_TERM}"

''' Functions for source terms '''

match SOURCE_TERM:
    case "NONE":
        source = lambda u, T, t: 0.0
    case "MMS1":
        assert N_DIMENSIONS == 1, "Dimensions for MMS1 must be 1"
        from modules.numerical.sources.s_mms_1 import source_1d as source
    case _:
        raise ValueError(f"Unknown source term: {SOURCE_TERM}")

#broken
@jit
def dudt(u, T, t):
    return source(u, T, t)