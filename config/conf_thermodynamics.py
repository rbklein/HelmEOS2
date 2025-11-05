"""
    Configuration for thermodynamic calculations.
"""

''' User-defined parameters '''

from data.molecules.carbondioxide import carbondioxide as molecule

NAME_MOLECULE = molecule.name

MOLAR_MASS = molecule.molar_mass #kg mol^-1

#Equation of State (EOS) type
EOS = "IDEAL_GAS" # Options: "IDEAL_GAS", "VAN_DER_WAALS", "PENG_ROBINSON", "WAGNER"

#Equation of State parameters
match EOS:
    case "IDEAL_GAS":
        EOS_parameters = molecule.ideal_gas_parameters
    case "VAN_DER_WAALS":
        EOS_parameters = molecule.Van_der_Waals_parameters
    case "PENG_ROBINSON":
        EOS_parameters = molecule.Peng_Robinson_parameters
    case "WAGNER":
        assert molecule.name == "CO_2", "Wagner equation is hard-coded for carbon dioxide" # CO_2 exclusive
        EOS_parameters = molecule.Wagner_parameters


#Dynamic viscosity
VISC_DYN = "CONSTANT"

VISC_DYN_parameters = {
    "value" : 0.0, #1e-5, #Pa s dynamic viscosity value 
}

#Bulk viscosity
VISC_BULK = "CONSTANT"

VISC_BULK_parameters = {
    "value" : 0.0, #bulk viscosity value
}

#Thermal conductivity
THERMAL_COND = "CONSTANT"

THERMAL_COND_parameters = {
    "value" :  0.0, #0.001, #thermal conductivity
}

''' Constants '''

UNIVERSAL_GAS_CONSTANT = 8.31446261815324 #J K^-1 mol^-1


