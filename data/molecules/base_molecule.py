"""
Abstract base class for molecules
"""

from abc import ABC, abstractmethod
from typing import Dict

class Molecule(ABC):
    @abstractmethod
    def molar_mass(self) -> float: pass

    @abstractmethod
    def ideal_gas_parameters(self) -> Dict[str, float]: pass

    @abstractmethod
    def Van_der_Waals_parameters(self) -> Dict[str, float]: pass

    #maybe also viscosities and thermal conductance functions 