"""
    Configuration file for uniform Cartesian grid parameters
"""

''' User-defined parameters '''

#Grid dimensions
N_DIMENSIONS = 3

PI = 3.141592653589793 # Also defined in conf_thermodynamics.py

#domain size (L_x, L_y, L_z), for 1D add comma
DOMAIN_SIZE = (2 * PI, 2 * PI, 2 * PI) # tuple of floats representing the size in each dimension

#grid resolution (n_x, n_y, n_z)
GRID_RESOLUTION = (64, 64, 64) # (128, 128, 128) # tuple of integers representing the number of grid cells in each dimension


