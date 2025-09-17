"""
    Configure initial and boundary conditions for a simulation.

    Later: add option to design the initial conditions in a more flexible way.

     Options: 
      - "DENSITY_WAVE_3D", 
      - "DENSITY_WAVE_2D", 
      - "TAYLOR_GREEN_VORTEX_3D", 
      - "CHAN_SHEAR_LAYER_2D", 
      - "PERIODIC_RICHMYER_MESHKOV_2D", 
      - "BLAST_WAVE_2D", 
      - "TURBULENCE_2D",
      - "COPPOLA_SHEAR_LAYER_2D",
      - "BERNADES_SHEAR_LAYER_2D"
"""

''' User-defined parameters '''

#Initial conditions
TEST_CASE = "BERNADES_SHEAR_LAYER_2D"

#Boundary condition type ([x_1, x_2], [y_1, y_2](, [z_1, z_2]))
BC_TYPES = [("PERIODIC", "PERIODIC"), ("PERIODIC", "PERIODIC")]

#Total simulation time
TOTAL_TIME = 0.1