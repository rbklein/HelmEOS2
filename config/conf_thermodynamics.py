"""
    Configuration for thermodynamic calculations.
"""

''' User-defined parameters '''

from data.molecules.nitrogen import nitrogen as molecule

MOLAR_MASS = molecule.molar_mass #kg mol^-1 (Nitrogen : N_2)

#Equation of State (EOS) type
EOS = "IDEAL_GAS" # Options: "IDEAL_GAS", "VAN_DER_WAALS"

#Equation of State parameters
match EOS:
    case "IDEAL_GAS":
        EOS_parameters = molecule.ideal_gas_parameters
    case "VAN_DER_WAALS":
        EOS_parameters = molecule.Van_der_Waals_parameters


#Dynamic viscosity
VISC_DYN = "CONSTANT"

VISC_DYN_parameters = {
    "value" : 0 #1e-5, #Pa s dynamic viscosity value 
}

#Bulk viscosity
VISC_BULK = "CONSTANT"

VISC_BULK_parameters = {
    "value" : 0.0, #bulk viscosity value
}

#Thermal conductivity
THERMAL_COND = "CONSTANT"

THERMAL_COND_parameters = {
    "value" : 0 #0.001, #thermal conductivity
}

''' Constants '''

UNIVERSAL_GAS_CONSTANT = 8.31446261815324 #J K^-1 mol^-1


