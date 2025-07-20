"""
Basis-Modul für alle Regelungstechnik-Module
"""

from abc import ABC, abstractmethod
import streamlit as st
from utils.calculation_logger import CalculationLogger

class BaseModule(ABC):
    """Basis-Klasse für alle Regelungstechnik-Module"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = CalculationLogger()
    
    @abstractmethod
    def render(self):
        """Rendert das Modul-Interface"""
        pass
    
    def display_description(self):
        """Zeigt die Modulbeschreibung an"""
        st.markdown(f"*{self.description}*")
        st.markdown("---")
    
    def reset_calculation(self):
        """Setzt die Berechnung zurück"""
        self.logger.clear()
        st.rerun()
