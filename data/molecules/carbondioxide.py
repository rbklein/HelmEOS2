"""
Carbon dioxide molecule
"""

from data.molecules.base_molecule import *

class CarbonDioxide(Molecule):
    """
        Data for CO_2 molecules
    """
    def __init__(self):
        self._molar_mass = 44.009e-3 #kg mol^-1 (Cabon dioxide CO_2)

        self._ideal_gas_parameters = {
            "gamma" : 1.4  # Specific heat ratio for ideal gas
        }

        self._Van_der_Waals_parameters = {
            "a_VdW" : 3.6e-1 / self._molar_mass**2,  # Van der Waals parameter a on mass basis (molar coeff / molar mass**2)
            "b_VdW": 4.267e-5 / self._molar_mass,   # Van der Waals parameter b on mass basis (molar coeff / molar mass)
            "molecular_dofs": 5,  # Degrees of freedom for the molecules (CO_2 is a linear molecule)
        }

        self._Peng_Robinson_parameters = {
            "acentric_factor" : 0.228, # Parameter used in Peng-Robinson (no units)
            "molecular_dofs": 5,  # Degrees of freedom for the molecules (CO_2 is a linear molecule)
        }

        self._rho_c = 467.7 #kg m^-3 critical density
        self._T_c = 304.128 #K critical temperature
        self._p_c = 7.3773e6 #Pa critical pressure
        

    @property
    def molar_mass(self): return self._molar_mass

    @property
    def ideal_gas_parameters(self): return self._ideal_gas_parameters
    
    @property
    def Van_der_Waals_parameters(self): return self._Van_der_Waals_parameters

    @property
    def Peng_Robinson_parameters(self): return self._Peng_Robinson_parameters

    @property
    def critical_points(self): return (self._rho_c, self._T_c, self._p_c)

#static instance
carbondioxide = CarbonDioxide()

if __name__ == "__main__":

    print(carbondioxide.molar_mass)
    print(carbondioxide.Van_der_Waals_parameters)