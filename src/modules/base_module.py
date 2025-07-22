"""
Basis-Modul für alle Regelungstechnik-Module
"""

from abc import ABC, abstractmethod
import streamlit as st
try:
    from src.utils.calculation_logger import CalculationLogger
    from src.utils.safe_sympify import safe_sympify
except ImportError:
    # Fallback wenn utils nicht verfügbar
    class CalculationLogger:
        def __init__(self):
            pass
        def log(self, *args, **kwargs):
            pass
    
    def safe_sympify(expr, symbols_dict=None):
        """Fallback safe_sympify function"""
        import sympy as sp
        import re
        if symbols_dict is None:
            symbols_dict = {}
        
        # Einfache automatische Multiplikation
        expr_str = str(expr)
        
        # Ersetze Muster wie "2s" mit "2*s"
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
        
        # Ersetze Muster wie "s(" mit "s*("
        expr_str = re.sub(r'([a-zA-Z])\(', r'\1*(', expr_str)
        
        # Ersetze Muster wie ")(" mit ")*(" für (s+1)(s+2) -> (s+1)*(s+2)
        expr_str = re.sub(r'\)\s*\(', r')*(', expr_str)
        
        # Ersetze Muster wie ")(s" mit ")*(s"
        expr_str = re.sub(r'\)([a-zA-Z])', r')*\1', expr_str)
        
        try:
            return sp.sympify(expr_str, locals=symbols_dict)
        except Exception:
            # Fallback: versuche es ohne Transformationen
            try:
                return sp.sympify(expr, locals=symbols_dict)
            except Exception:
                raise ValueError(f"Konnte '{expr}' nicht parsen. Versuchen Sie explizite Multiplikation (*) zu verwenden.")

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
