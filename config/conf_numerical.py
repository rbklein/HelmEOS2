"""
    Configuration file for numerical methods.
"""

''' User-defined parameters '''

#Numerical flux type
NUMERICAL_FLUX = "KEEP_DG"

#Numerical viscous flux type
NUMERICAL_VISCOUS_FLUX = "NAIVE"

#Numerical heat flux type
NUMERICAL_HEAT_FLUX = "NAIVE"

#Time-stepping method
TIME_STEP_METHOD = "RK4"

#Discrete gradient type
DISCRETE_GRADIENT = "GONZALEZ"

#Number of time steps
NUM_TIME_STEPS = 100000

