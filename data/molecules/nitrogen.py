"""
Nitrogen molecule
"""

from data.molecules.base_molecule import *

class Nitrogen(Molecule):
    """
        Data for N_2 molecules
    """
    def __init__(self):
        self._molar_mass = 28.01348e-3 #kg mol^-1 (Nitrogen : N_2)

        self._ideal_gas_parameters = {
            "gamma" : 1.4  # Specific heat ratio for ideal gas
        }

        self._Van_der_Waals_parameters = {
            #"a_VdW" : 1.37e-1 / self._molar_mass**2,  # Van der Waals parameter a on mass basis (molar coeff / molar mass**2)
            #"b_VdW": 3.87e-5 / self._molar_mass,   # Van der Waals parameter b on mass basis (molar coeff / molar mass)
            "molecular_dofs": 5,  # Degrees of freedom for the molecules
        }

        self._Peng_Robinson_parameters = {
            "acentric_factor" : 0.0372, # Parameter used in Peng-Robinson (no units)
            "molecular_dofs": 5,  # Degrees of freedom for the molecules
        }

        # residual Helmholtz coefficients not implemented yet
        self._Wagner_parameters = { 
            "T_min" : 63,  # Minimal valid temperature 
            "T_max" : 700, # Maximal valid temperature
            "p_max" : 300e6 # Maximal valid pressure                            
        }

        self._rho_c = 11.1839 * self._molar_mass #kg m^-3 critical density
        self._T_c = 126.192 #K critical temperature
        self._p_c = 3.3958e6 #Pa critical pressure

        self._name = "N_2"

    @property
    def molar_mass(self): return self._molar_mass

    @property
    def ideal_gas_parameters(self): return self._ideal_gas_parameters
    
    @property
    def Van_der_Waals_parameters(self): return self._Van_der_Waals_parameters

    @property
    def Peng_Robinson_parameters(self): return self._Peng_Robinson_parameters

    @property
    def critical_point(self): return (self._rho_c, self._T_c, self._p_c)

    @property
    def name(self): return self._name

#static instance
nitrogen = Nitrogen()

if __name__ == "__main__":

    print(nitrogen.molar_mass)
    print(nitrogen.Van_der_Waals_parameters)