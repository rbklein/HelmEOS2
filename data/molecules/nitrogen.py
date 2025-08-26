"""
Nitrogen molecule
"""

from data.molecules.base_molecule import *

class Nitrogen(Molecule):
    """
        Data for N_2 molecules
    """
    def __init__(self):
        self._molar_mass = 28.0134e-3 #kg mol^-1 (Nitrogen : N_2)

        self._ideal_gass_parameters = {
            "gamma" : 1.4  # Specific heat ratio for ideal gas
        }

        self._Van_der_Waals_parameters = {
            "a_VdW" : 1.37e-1 / self._molar_mass**2,  # Van der Waals parameter a on mass basis (molar coeff / molar mass**2)
            "b_VdW": 3.87e-5 / self._molar_mass,   # Van der Waals parameter b on mass basis (molar coeff / molar mass)
            "molecular_dofs": 5,  # Degrees of freedom for the molecules
        }

    @property
    def molar_mass(self): return self._molar_mass

    @property
    def ideal_gas_parameters(self): return self._ideal_gass_parameters
    
    @property
    def Van_der_Waals_parameters(self): return self._Van_der_Waals_parameters

#static instance
nitrogen = Nitrogen()

if __name__ == "__main__":

    print(nitrogen.molar_mass)
    print(nitrogen.Van_der_Waals_parameters)