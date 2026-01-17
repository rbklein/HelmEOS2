"""
    Configure initial and boundary conditions for a simulation.

    Later: add option to design the initial conditions in a more flexible way.

     Options: 
      - "DENSITY_WAVE_1D",
      - "TAYLOR_GREEN_VORTEX_3D", 
      - "CHAN_SHEAR_LAYER_2D",  
      - "TURBULENCE_2D",
      - "GRESHO_VORTEX",

     Manufactured solution option:
      - "MMS1_1D"
"""

''' User-defined parameters '''

#Initial conditions
TEST_CASE = "TAYLOR_GREEN_VORTEX_3D"

#Boundary condition type ([x_1, x_2], [y_1, y_2](, [z_1, z_2]))
BC_TYPES = [("PERIODIC", "PERIODIC"), ("PERIODIC", "PERIODIC"), ("PERIODIC", "PERIODIC")]

_num_conv_times = 20
_conv_time      = 1.432335668626272e-07

#Total simulation time
TOTAL_TIME = _conv_time * _num_conv_times