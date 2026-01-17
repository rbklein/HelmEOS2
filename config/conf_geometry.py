"""
    Configuration file for uniform Cartesian grid parameters
"""

''' User-defined parameters '''

#Grid dimensions
N_DIMENSIONS = 3

PI = 3.141592653589793 # Also defined in conf_thermodynamics.py
L = 1.3751829634648355e-06 # 4.125548890394506e-06

#domain size (L_x, L_y, L_z), for 1D add comma
DOMAIN_SIZE = (2 * PI * L, 2 * PI * L, 2 * PI * L) # tuple of floats representing the size in each dimension

#grid resolution (n_x, n_y, n_z)
GRID_RESOLUTION = (512, 512, 512) # tuple of integers representing the number of grid cells in each dimension




