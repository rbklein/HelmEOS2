"""
    Setup initial conditions for the simulation.
"""

from config.conf_geometry   import *
from config.conf_simulation import *

''' Consistency checks '''

KNOWN_TEST_CASES = [
    "DENSITY_WAVE_1D", 
    "TAYLOR_GREEN_VORTEX_3D", 
    "CHAN_SHEAR_LAYER_2D", 
    "TURBULENCE_2D",
    "MMS1_1D"
]

assert TEST_CASE in KNOWN_TEST_CASES, f"Unknown test case: {TEST_CASE}. Known test cases are: {KNOWN_TEST_CASES}"

''' Functions for initial conditions '''


match TEST_CASE: 
    case "DENSITY_WAVE_1D":
        
        ''' set initial condition for 1D density wave '''

        # Ensure the number of dimensions is 1 for this test case
        assert N_DIMENSIONS == 1, "DENSITY_WAVE_1D requires 1 dimensions"
        
        from modules.simulation.initial_conditions.density_wave import density_wave_1d as initial_condition
    case "TAYLOR_GREEN_VORTEX_3D":
        
        ''' set initial condition for 3D Taylor-Green vortex '''

        # Ensure the number of dimensions is 3 for this test case
        assert N_DIMENSIONS == 3, "TAYLOR_GREEN_VORTEX_3D requires 3 dimensions"

        from modules.simulation.initial_conditions.taylor_green import Taylor_Green_vortex_3d as initial_condition
    case "CHAN_SHEAR_LAYER_2D":

        ''' set initial condition for 2D supercritical shear layer '''

        # Ensure the number of dimensions is 2 for this test case
        assert N_DIMENSIONS == 2, "SHEAR_LAYER_2D requires 2 dimensions"

        from modules.simulation.initial_conditions.shear_layer import chan_shear_layer_2d as initial_condition
    case "TURBULENCE_2D":

        ''' set initial condition for 2D turbulence '''
        # Ensure the number of dimensions is 2 for this test case
        assert N_DIMENSIONS == 2, "TURBULENCE_2D requires 2 dimensions"
        #from modules.simulation.initial_conditions.turbulence2D import turbulence_2d as initial_condition
        NotImplementedError("Turbulence 2D out of order")
    case "MMS1_1D":
        ''' set initial condition for manufactured solution 1 '''

        # Ensure 1D and correct source term
        from config.conf_numerical import SOURCE_TERM
        assert N_DIMENSIONS == 1, "MMS1_1D requires 1 dimension"
        assert SOURCE_TERM == "MMS1", "MMS1_1D requires compatible source term MMS1"

        from modules.simulation.initial_conditions.u_mms_1 import u_mms_1 as initial_condition
    case _:
        raise ValueError(f"Unknown test case: {TEST_CASE}")