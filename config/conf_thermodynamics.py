"""
    Configuration for thermodynamic calculations.
"""

''' User-defined parameters '''

from data.molecules.carbondioxide import carbondioxide as molecule

NAME_MOLECULE = molecule.name

MOLAR_MASS = molecule.molar_mass #kg mol^-1

# Equation of State (EOS) type
# Options: "IDEAL_GAS", "VAN_DER_WAALS", "PENG_ROBINSON", "KUNZ_WAGNER", "KUNZ_WAGNER"
EOS = "KUNZ_WAGNER" 

# Equation of State parameters
match EOS:
    case "IDEAL_GAS":
        EOS_parameters = molecule.ideal_gas_parameters
    case "VAN_DER_WAALS":
        EOS_parameters = molecule.Van_der_Waals_parameters
    case "PENG_ROBINSON":
        EOS_parameters = molecule.Peng_Robinson_parameters
    case "KUNZ_WAGNER":
        assert molecule.name == "CO_2", "Wagner equation is hard-coded for carbon dioxide" # CO_2 exclusive
        EOS_parameters = molecule.Wagner_parameters
    case "KUNZ_WAGNER_MANUAL":
        assert molecule.name == "CO_2", "Wagner equation is hard-coded for carbon dioxide" # CO_2 exclusive
        EOS_parameters = molecule.Wagner_parameters


# Dynamic viscosity 
# Options: "CONSTANT", "LAESECKE"
VISC_DYN = "LAESECKE"

VISC_DYN_parameters = { 
    'value' : 1e-4
}

# Bulk viscosity
VISC_BULK = "CONSTANT"

VISC_BULK_parameters = {
    "value" : 0.0, #bulk viscosity value
}

#Thermal conductivity
# Options: "CONSTANT", "HUBER"
THERMAL_COND = "HUBER"

THERMAL_COND_parameters = {
    'value' : 1e-4
 }

''' Constants '''

UNIVERSAL_GAS_CONSTANT = 8.31446261815324 #J K^-1 mol^-1
AVOGADRO_CONSTANT = 6.02214085774e23 #mol^-1
BOLTZMANN_CONSTANT = UNIVERSAL_GAS_CONSTANT / AVOGADRO_CONSTANT #J K^-1
R_specific = UNIVERSAL_GAS_CONSTANT / MOLAR_MASS #J kg^-1 K^-1

PI = 3.141592653589793 #Also defined in conf_geometry.py