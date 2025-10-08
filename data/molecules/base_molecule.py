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

    @abstractmethod
    def Peng_Robinson_parameters(self) -> Dict[str, float]: pass

    @abstractmethod
    def critical_points(self) -> tuple[float, float, float]: pass

    @abstractmethod
    def name(self) -> str: pass

    #maybe also viscosities and thermal conductance functions 