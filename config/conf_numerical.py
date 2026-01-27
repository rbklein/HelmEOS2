"""
    Configuration file for numerical methods.
"""

''' User-defined parameters '''

#Numerical flux type
NUMERICAL_FLUX = "AIELLO"

#Numerical viscous flux type
NUMERICAL_VISCOUS_FLUX = "NAIVE"

#Numerical heat flux type
NUMERICAL_HEAT_FLUX = "NAIVE"

#Numerical source term
SOURCE_TERM = "NONE"

#Time-stepping method
TIME_STEP_METHOD = "RK4"

#Discrete gradient type
#Options: "SYM_ITOH_ABE", "GONZALEZ"
DISCRETE_GRADIENT = "SYM_ITOH_ABE"

_num_conv_times = 50
_num_steps_per_conv = 256 * 25

#Number of time steps
NUM_TIME_STEPS = _num_steps_per_conv * _num_conv_times

