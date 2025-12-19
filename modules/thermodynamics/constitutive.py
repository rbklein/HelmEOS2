"""
    Functions constructing constitutive laws used in the computation of irreversible processes
"""

from prep_jax import *
from config.conf_thermodynamics import *

''' Consistency checks '''

KNOWN_VISC_DYN = ["CONSTANT", "LAESECKE"]

assert VISC_DYN in KNOWN_VISC_DYN, f"Dynamic viscosity {VISC_DYN} not known"
match VISC_DYN:
    case "CONSTANT":
        from modules.thermodynamics.dynamic.constant_dynamic import check_consistency as check_consistency_dynamic
    case "LAESECKE":
        from modules.thermodynamics.dynamic.laesecke_dynamic import check_consistency as check_consistency_dynamic

KNOWN_VISC_BULK = ["CONSTANT"]

assert VISC_BULK in KNOWN_VISC_BULK, f"Bulk viscosity {VISC_BULK} not known"
if VISC_BULK == "CONSTANT":
    from modules.thermodynamics.bulk.constant_bulk import check_consistency as check_consistency_bulk

KNOWN_THERMAL_COND = ["CONSTANT"]

assert THERMAL_COND in KNOWN_THERMAL_COND, f"Thermal conductivity {THERMAL_COND} not known"
if THERMAL_COND == "CONSTANT":
    from modules.thermodynamics.conduction.constant_conductivity import check_consistency as check_consistency_thermal

check_consistency_dynamic()
check_consistency_bulk()
check_consistency_thermal()

''' Functions '''

match VISC_DYN:
    case "CONSTANT":
        from modules.thermodynamics.dynamic.constant_dynamic import constant_dynamic_viscosity as dynamic_viscosity
    case "LAESECKE":
        from modules.thermodynamics.dynamic.laesecke_dynamic import laesecke_dynamic_viscosity as dynamic_viscosity
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



