"""
    Configuration for post-processing steps in simulation pipeline.

    quantities that can be plotted:
    - density
    - momentum
    - total energy
    - pressure
    - pressure fluctuations
    - temperature
    - entropy
    - internal energy
    - helmholtz energy
    - velocity
    - vorticity
    X - reduced distance to critical point (removed due to ambiguity between molecule critical point and EOS critical point)
    - speed of sound
    - local Mach number

    integrals that can be plotted:
    - entropy
    
    other plots:
    - energy spectrum

"""

# sequence of post-processing steps to be applied
PLOT_SEQUENCE = [
    "DENSITY",
    "VELOCITY",
    "TEMPERATURE",
    "PRESSURE",
    "LOCAL_MACH",
    "ENTROPY",
]

#TO DO: entropy production (scalar field) evolution over time same for kinetic energy 

MAX_PLOTS_PER_ROW = 3  # Maximum number of plots per row in the output figure

COLORMAP = 'magma'
MAX_TIME_SERIES_LENGTH = 30
NUM_ITS_PER_UPDATE = 10

# Determine index of slice to plot for a 3D simulation
# Could be improved by only computing desire values for the slice instead of on the whole grid
SLICE_3D = (32//2, slice(None), slice(None))
