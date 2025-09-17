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
    - velocity
    - vorticity
    - reduced distance to critical point
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
    "PRESSURE_FLUCTUATIONS",
    "VELOCITY",
    "VORTICITY",
    "LOCAL_MACH",
    "SPEED_OF_SOUND",
]

#TO DO: entropy production (scalar field) evolution over time same for kinetic energy 

MAX_PLOTS_PER_ROW = 3  # Maximum number of plots per row in the output figure

COLORMAP = 'magma'
MAX_TIME_SERIES_LENGTH = 30
NUM_ITS_PER_UPDATE = 100