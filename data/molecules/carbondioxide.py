"""
Carbon dioxide molecule
"""

from data.molecules.base_molecule import *

class CarbonDioxide(Molecule):
    """
        Data for CO_2 molecules
    """
    def __init__(self):
        self._molar_mass = 44.0098e-3 #kg mol^-1 (Cabon dioxide CO_2)

        self._ideal_gas_parameters = {
            "gamma" : 1.4  # Specific heat ratio for ideal gas
        }

        self._Van_der_Waals_parameters = {
            #"a_VdW" : 3.6565212264615676e-1 / self._molar_mass**2,  # Van der Waals parameter a on mass basis (molar coeff / molar mass**2)
            #"b_VdW": 4.284532535660459e-05 / self._molar_mass,   # Van der Waals parameter b on mass basis (molar coeff / molar mass)
            "molecular_dofs": 5,  # Degrees of freedom for the molecules (CO_2 is a linear molecule)
        }

        self._Peng_Robinson_parameters = {
            "acentric_factor" : 0.22394, # Parameter used in Peng-Robinson (no units)
            "molecular_dofs": 5,  # Degrees of freedom for the molecules (CO_2 is a linear molecule)
        }

        self._Wagner_parameters = { }

        self._rho_c = 10.6249e3 * self._molar_mass #kg m^-3 critical density
        self._T_c = 304.1282 #K critical temperature
        self._p_c = 7.3773e6 #Pa critical pressure

        self._name = "CO_2"
        

    @property
    def molar_mass(self): return self._molar_mass

    @property
    def ideal_gas_parameters(self): return self._ideal_gas_parameters
    
    @property
    def Van_der_Waals_parameters(self): return self._Van_der_Waals_parameters

    @property
    def Peng_Robinson_parameters(self): return self._Peng_Robinson_parameters

    #Exclusively for Carbon Dioxide
    @property
    def Wagner_parameters(self): return self._Wagner_parameters

    @property
    def critical_points(self): return (self._rho_c, self._T_c, self._p_c)

    @property
    def name(self): return self._name

#static instance
carbondioxide = CarbonDioxide()

if __name__ == "__main__":

    print(carbondioxide.molar_mass)
    print(carbondioxide.Van_der_Waals_parameters)