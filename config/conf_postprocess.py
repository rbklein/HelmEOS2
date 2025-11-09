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
    "MOMENTUM",
    "TOTAL_ENERGY",
    "TEMPERATURE",
    "PRESSURE",
    "ENTROPY",
    "INTERNAL_ENERGY",
    "HELMHOLTZ"
]

#TO DO: entropy production (scalar field) evolution over time same for kinetic energy 

MAX_PLOTS_PER_ROW = 4  # Maximum number of plots per row in the output figure

COLORMAP = 'magma'
MAX_TIME_SERIES_LENGTH = 30
NUM_ITS_PER_UPDATE = 100