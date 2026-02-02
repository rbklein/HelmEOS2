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
TEST_CASE = "DENSITY_WAVE_1D"

#Boundary condition type ([x_1, x_2], [y_1, y_2](, [z_1, z_2]))
BC_TYPES = [("PERIODIC", "PERIODIC")]

# _num_conv_times = 20 # 50
# _conv_time      = 5.4610594533757567e-05 # 1.432335668626272e-07 # 0.0074666290108899355 # 1.5914840762514136e-08 #

#Total simulation time
TOTAL_TIME = 0.5