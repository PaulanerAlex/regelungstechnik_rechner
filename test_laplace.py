#!/usr/bin/env python3
"""
Test-Umgebung für Laplace-Transformation
Testet verschiedene Funktionen und identifiziert Probleme
"""

import sympy as sp
import sys
import os

# Pfad zum src-Verzeichnis hinzufügen
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from modules.laplace_transform import LaplaceTransformModule
from utils.calculation_logger import CalculationLogger

def test_sympy_laplace():
    """Teste SymPy's eingebaute Laplace-Transformation"""
    print("=== SymPy Laplace-Transformation Tests ===")
    
    t, s = sp.symbols('t s', real=True, positive=True)
    a = sp.Symbol('a', real=True, positive=True)
    
    test_functions = [
        ("exp(-2*t)", sp.exp(-2*t)),
        ("exp(-a*t)", sp.exp(-a*t)),
        ("exp(-t)", sp.exp(-t)),
        ("sin(3*t)", sp.sin(3*t)),
        ("cos(3*t)", sp.cos(3*t)),
        ("t", t),
        ("t**2", t**2),
        ("exp(-2*t)*cos(3*t)", sp.exp(-2*t)*sp.cos(3*t)),
        ("exp(-a*t)*sin(3*t)", sp.exp(-a*t)*sp.sin(3*t))
    ]
    
    for name, func in test_functions:
        print(f"\n--- Test: {name} ---")
        try:
            result = sp.laplace_transform(func, t, s)
            if isinstance(result, tuple):
                F_s = result[0]
                print(f"✓ SymPy Ergebnis: {F_s}")
                
                # Prüfe auf bekannte Probleme
                if F_s.has(t):
                    print(f"⚠️  PROBLEM: Ergebnis enthält noch 't': {F_s}")
                elif F_s == func/s:
                    print(f"⚠️  PROBLEM: Triviales f(t)/s Ergebnis: {F_s}")
                elif str(F_s) == str(s):
                    print(f"⚠️  PROBLEM: Ergebnis ist nur 's': {F_s}")
                else:
                    print(f"✓ Ergebnis scheint korrekt")
            else:
                print(f"✓ SymPy Ergebnis: {result}")
        except Exception as e:
            print(f"✗ SymPy Fehler: {e}")

def test_manual_laplace():
    """Teste die manuelle Laplace-Transformation"""
    print("\n\n=== Manuelle Laplace-Transformation Tests ===")
    
    # Instanz erstellen
    module = LaplaceTransformModule()
    module.logger = CalculationLogger()
    
    t, s = sp.symbols('t s', real=True, positive=True)
    a = sp.Symbol('a', real=True, positive=True)
    
    test_functions = [
        ("exp(-2*t)", sp.exp(-2*t)),
        ("exp(-a*t)", sp.exp(-a*t)),
        ("exp(-t)", sp.exp(-t)),
        ("sin(3*t)", sp.sin(3*t)),
        ("cos(3*t)", sp.cos(3*t)),
        ("t", t),
        ("t**2", t**2),
        ("exp(-2*t)*cos(3*t)", sp.exp(-2*t)*sp.cos(3*t)),
        ("exp(-a*t)*sin(3*t)", sp.exp(-a*t)*sp.sin(3*t)),
        ("1", 1)
    ]
    
    for name, func in test_functions:
        print(f"\n--- Test: {name} ---")
        try:
            result = module._manual_laplace_transform(func, t, s)
            print(f"✓ Manuelles Ergebnis: {result}")
            
            # Prüfe Typ des Ergebnisses
            if hasattr(result, 'func') and result.func == sp.LaplaceTransform:
                print(f"ℹ️  Symbolisches Ergebnis (kein Pattern gefunden)")
            else:
                print(f"✓ Direktes Ergebnis gefunden")
                
        except Exception as e:
            print(f"✗ Manueller Fehler: {e}")
            import traceback
            traceback.print_exc()

def test_combined():
    """Teste die kombinierte Transformation (wie im echten Code)"""
    print("\n\n=== Kombinierte Transformation Tests ===")
    
    module = LaplaceTransformModule()
    module.logger = CalculationLogger()
    
    test_inputs = [
        "exp(-2*t)",
        "exp(-a*t)", 
        "exp(-t)",
        "sin(3*t)",
        "cos(3*t)",
        "t",
        "exp(-2*t)*cos(3*t)",
        "exp(-a*t)*sin(3*t)"
    ]
    
    for func_str in test_inputs:
        print(f"\n--- Test: {func_str} ---")
        try:
            # Simuliere den echten Workflow
            t, s = sp.symbols('t s', real=True, positive=True)
            f_t = sp.sympify(func_str)
            
            # SymPy versuchen
            try:
                laplace_result = sp.laplace_transform(f_t, t, s)
                if isinstance(laplace_result, tuple):
                    F_s_result = laplace_result[0]
                else:
                    F_s_result = laplace_result
                
                # Prüfung wie im echten Code
                needs_manual = (F_s_result.has(t) or 
                               F_s_result == s or 
                               str(F_s_result) == str(s) or
                               F_s_result == f_t/s)
                
                if needs_manual:
                    print(f"⚠️  SymPy fehlerhaft: {F_s_result} -> verwende manuell")
                    F_s_final = module._manual_laplace_transform(f_t, t, s)
                    print(f"✓ Manuelles Ergebnis: {F_s_final}")
                else:
                    print(f"✓ SymPy Ergebnis OK: {F_s_result}")
                    F_s_final = F_s_result
                    
            except Exception as e:
                print(f"✗ SymPy komplett fehlgeschlagen: {e}")
                F_s_final = module._manual_laplace_transform(f_t, t, s)
                print(f"✓ Manuelles Fallback: {F_s_final}")
                
        except Exception as e:
            print(f"✗ Kompletter Fehler: {e}")
            import traceback
            traceback.print_exc()

def test_string_patterns():
    """Teste die String-Pattern-Erkennung"""
    print("\n\n=== String-Pattern Tests ===")
    
    import re
    
    test_strings = [
        "exp(-2*t)",
        "exp(-a*t)",
        "exp(-3*t)",
        "exp(-t)",
        "sin(3*t)",
        "cos(2*t)",
        "sin(t)",
        "cos(t)"
    ]
    
    for test_str in test_strings:
        print(f"\n--- Pattern Test: {test_str} ---")
        
        # exp Pattern
        exp_pattern = re.match(r'exp\((-?\w*\*?t)\)', test_str)
        if exp_pattern:
            arg = exp_pattern.group(1)
            print(f"✓ Exp Pattern erkannt: {arg}")
        
        # sin Pattern  
        sin_pattern = re.match(r'sin\((\w*\*?t)\)', test_str)
        if sin_pattern:
            arg = sin_pattern.group(1)
            print(f"✓ Sin Pattern erkannt: {arg}")
            
        # cos Pattern
        cos_pattern = re.match(r'cos\((\w*\*?t)\)', test_str)
        if cos_pattern:
            arg = cos_pattern.group(1)
            print(f"✓ Cos Pattern erkannt: {arg}")

if __name__ == "__main__":
    print("Laplace-Transformation Testumgebung")
    print("=" * 50)
    
    test_sympy_laplace()
    test_manual_laplace()
    test_combined()
    test_string_patterns()
    
    print("\n\n=== Test Zusammenfassung ===")
    print("Prüfe die Ausgabe oben auf ⚠️  und ✗ Markierungen für Probleme")
