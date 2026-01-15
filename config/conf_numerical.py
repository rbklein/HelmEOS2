"""
    Configuration file for numerical methods.
"""

''' User-defined parameters '''

#Numerical flux type
NUMERICAL_FLUX = "CHANDRASHEKAR_IDEAL"

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

#Number of time steps
NUM_TIME_STEPS = 50000

