"""
    Functions constructing constitutive laws used in the computation of irreversible processes
"""

from prep_jax import *
from config.conf_thermodynamics import *

''' Consistency checks '''

KNOWN_VISC_DYN = ["CONSTANT"]

assert VISC_DYN in KNOWN_VISC_DYN, f"Dynamic viscosity {VISC_DYN} not known"
if VISC_DYN == "CONSTANT":
    assert "value" in VISC_DYN_parameters, "Constant dynamic viscosity must be assigned a value"  
    assert isinstance(VISC_DYN_parameters["value"], (float, int)), "Constant dynamic viscosity value must be a floating point value or an integer"
    assert VISC_DYN_parameters["value"] >= 0.0, "Constant dynamic viscosity must be non-negative"

KNOWN_VISC_BULK = ["CONSTANT"]

assert VISC_BULK in KNOWN_VISC_BULK, f"Bulk viscosity {VISC_BULK} not known"
if VISC_BULK == "CONSTANT":
    assert "value" in VISC_BULK_parameters, "Constant bulk viscosity must be assigned a value"
    assert isinstance(VISC_BULK_parameters["value"], (float, int)), "Constant bulk viscosity value must be a floating point value or an integer"
    assert VISC_BULK_parameters["value"] >= 0.0, "Constant bulk viscosity must be non-negative"

KNOWN_THERMAL_COND = ["CONSTANT"]

assert THERMAL_COND in KNOWN_THERMAL_COND, f"Thermal conductivity {THERMAL_COND} not known"
if THERMAL_COND == "CONSTANT":
    assert "value" in THERMAL_COND_parameters, "Constant thermal conductivity must be assigned a value"
    assert isinstance(THERMAL_COND_parameters["value"], (float, int)), "Constant thermal conductivity value must be a floating point value or an integer"
    assert THERMAL_COND_parameters["value"] >= 0.0, "Constant bulk viscosity must be non-negative"


''' Functions '''

match VISC_DYN:
    case "CONSTANT":
        from modules.thermodynamics.dynamic.constant_dynamic import constant_dynamic_viscosity as dynamic_viscosity
    case _:
        raise ValueError(f"Unknown dynamic viscosity function: {VISC_DYN}")

match VISC_BULK:
    case "CONSTANT":
        from modules.thermodynamics.bulk.constant_bulk import constant_bulk_viscosity as bulk_viscosity
    case _:
        raise ValueError(f"Unknown bulk viscosity function: {VISC_BULK}")

match THERMAL_COND:
    case "CONSTANT":
        from modules.thermodynamics.conduction.constant_conductivity import constant_thermal_conductivity as thermal_conductivity
    case _:
        raise ValueError(f"Unknown thermal conductivity function: {THERMAL_COND}")



