"""
    Configuration file for numerical methods.
"""

''' User-defined parameters '''

#Numerical flux type
NUMERICAL_FLUX = "KEEP"

#Numerical viscous flux type
NUMERICAL_VISCOUS_FLUX = "NAIVE"

#Numerical heat flux type
NUMERICAL_HEAT_FLUX = "NAIVE"

#Numerical source term
SOURCE_TERM = "NONE"

#Time-stepping method
TIME_STEP_METHOD = "RK4"

#Discrete gradient type
DISCRETE_GRADIENT = "SYM_ITOH_ABE"

#Number of time steps
NUM_TIME_STEPS = 10 # 6250 # 12500 #

