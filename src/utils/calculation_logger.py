"""
Calculation Logger für nachvollziehbare Rechenwege
"""

from typing import List, Dict, Any, Optional
import sympy as sp

class CalculationLogger:
    """Logger für schrittweise Berechnungen"""
    
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.current_step = 0
    
    def add_step(self, description: str, expression: Optional[sp.Basic] = None, 
                 explanation: str = "", code: str = ""):
        """Fügt einen Berechnungsschritt hinzu"""
        step = {
            "step_number": self.current_step + 1,
            "description": description,
            "explanation": explanation
        }
        
        if expression is not None:
            step["expression"] = expression
        
        if code:
            step["code"] = code
            
        self.steps.append(step)
        self.current_step += 1
    
    def add_explanation(self, text: str):
        """Fügt eine Erklärung zum aktuellen Schritt hinzu"""
        if self.steps:
            self.steps[-1]["explanation"] = text
    
    def add_latex_step(self, description: str, latex_expression: str, explanation: str = ""):
        """Fügt einen Schritt mit LaTeX-Expression hinzu"""
        step = {
            "step_number": self.current_step + 1,
            "description": description,
            "latex": latex_expression,
            "explanation": explanation
        }
        self.steps.append(step)
        self.current_step += 1
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """Gibt alle Schritte zurück"""
        return self.steps
    
    def clear(self):
        """Löscht alle Schritte"""
        self.steps = []
        self.current_step = 0
    
    def get_last_step(self) -> Optional[Dict[str, Any]]:
        """Gibt den letzten Schritt zurück"""
        return self.steps[-1] if self.steps else None
