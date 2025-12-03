"""
    Configure initial and boundary conditions for a simulation.

    Later: add option to design the initial conditions in a more flexible way.

     Options: 
      - "DENSITY_WAVE_1D",
      - "TAYLOR_GREEN_VORTEX_3D", 
      - "CHAN_SHEAR_LAYER_2D",  
      - "TURBULENCE_2D",

     Manufactured solution option:
      - "MMS1_1D"
"""

''' User-defined parameters '''

#Initial conditions
TEST_CASE = "MMS1_1D"

#Boundary condition type ([x_1, x_2], [y_1, y_2](, [z_1, z_2]))
BC_TYPES = [("PERIODIC", "PERIODIC")]

#Total simulation time
TOTAL_TIME = 1e-5
