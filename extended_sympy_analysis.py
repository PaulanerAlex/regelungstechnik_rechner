#!/usr/bin/env python3
"""
Extended SymPy bug analysis - tests different approaches and edge cases
Erweiterte Analyse des SymPy-Bugs mit verschiedenen Ansätzen
"""

import sympy as sp

def test_different_input_formats():
    """Teste verschiedene Eingabeformate für die gleiche Funktion"""
    print("=" * 60)
    print("TEST VERSCHIEDENER EINGABEFORMATE")
    print("=" * 60)
    
    t, s = sp.symbols('t s', real=True, positive=True)
    
    # Verschiedene Wege, die gleiche Funktion zu definieren
    input_variants = [
        "2*t*exp(-4*t)",           # String
        2*t*sp.exp(-4*t),          # Direkte SymPy-Objekte
        t*sp.exp(-4*t)*2,          # Umgestellte Reihenfolge
        sp.Mul(2, t, sp.exp(-4*t)), # Explizite Multiplikation
        "t*exp(-4*t)*2",           # String mit anderer Reihenfolge
    ]
    
    expected = 2/(s+4)**2
    
    for i, variant in enumerate(input_variants):
        print(f"Variante {i+1}: {variant}")
        
        # Parse if string
        if isinstance(variant, str):
            f_t = sp.sympify(variant)
        else:
            f_t = variant
            
        print(f"  Parsed: {f_t}")
        
        # Test Laplace transform
        try:
            result = sp.laplace_transform(f_t, t, s)
            F_s = result[0] if isinstance(result, tuple) else result
            
            print(f"  SymPy: {F_s}")
            
            # Check for bug
            if F_s == f_t/s or str(F_s) == f"{f_t}/s":
                print("  🐛 BUG: f(t)/s Ergebnis")
            elif sp.simplify(F_s - expected) == 0:
                print("  ✅ KORREKT!")
            else:
                print("  ⚠️  Anderes Ergebnis")
                
        except Exception as e:
            print(f"  ❌ Fehler: {e}")
        
        print()


def test_edge_cases():
    """Teste Grenzfälle und verwandte Funktionen"""
    print("=" * 60)
    print("TEST GRENZFÄLLE UND VERWANDTE FUNKTIONEN")
    print("=" * 60)
    
    t, s = sp.symbols('t s', real=True, positive=True)
    
    test_cases = [
        # Basis-Exponentialfunktionen (sollten funktionieren)
        ("exp(-4*t)", "1/(s+4)"),
        ("2*exp(-4*t)", "2/(s+4)"),
        
        # Potenzfunktionen (sollten funktionieren)
        ("t", "1/s**2"),
        ("2*t", "2/s**2"),
        ("t**2", "2/s**3"),
        
        # Problematische t*exp(-a*t) Fälle
        ("t*exp(-t)", "1/(s+1)**2"),
        ("t*exp(-4*t)", "1/(s+4)**2"),
        ("2*t*exp(-4*t)", "2/(s+4)**2"),
        
        # Höhere Potenzen t^n*exp(-a*t)
        ("t**2*exp(-t)", "2/(s+1)**3"),
        ("t**3*exp(-2*t)", "6/(s+2)**4"),
    ]
    
    for func_str, expected_str in test_cases:
        print(f"Test: {func_str}")
        
        f_t = sp.sympify(func_str)
        expected = sp.sympify(expected_str)
        
        try:
            result = sp.laplace_transform(f_t, t, s)
            F_s = result[0] if isinstance(result, tuple) else result
            
            print(f"  Erwartet: {expected}")
            print(f"  SymPy:    {F_s}")
            
            # Prüfe Korrektheit
            if F_s == f_t/s:
                print("  🐛 BUG: f(t)/s")
            elif sp.simplify(F_s - expected) == 0:
                print("  ✅ KORREKT")
            else:
                diff = sp.simplify(F_s - expected)
                print(f"  ⚠️  ABWEICHUNG: {diff}")
                
        except Exception as e:
            print(f"  ❌ FEHLER: {e}")
        
        print()


def test_manual_integration():
    """Teste manuelle Integration zur Verifikation"""
    print("=" * 60)
    print("MANUELLE INTEGRATION ZUR VERIFIKATION")
    print("=" * 60)
    
    t, s = sp.symbols('t s', real=True, positive=True)
    
    print("Berechnung von L{2*t*exp(-4*t)} durch Definition:")
    print("L{f(t)} = ∫₀^∞ f(t)·e^(-st) dt")
    print()
    
    # Definiere die Funktion
    f_t = 2*t*sp.exp(-4*t)
    print(f"f(t) = {f_t}")
    
    # Laplace-Integral: ∫₀^∞ f(t)·e^(-st) dt
    integrand = f_t * sp.exp(-s*t)
    print(f"Integrand: f(t)·e^(-st) = {integrand}")
    print(f"Vereinfacht: {sp.simplify(integrand)}")
    print()
    
    print("Berechne: ∫₀^∞ 2t·e^(-(s+4)t) dt")
    
    try:
        # Manuelle Integration
        integral_result = sp.integrate(integrand, (t, 0, sp.oo))
        print(f"Manuelles Integral: {integral_result}")
        
        # Vereinfache
        simplified = sp.simplify(integral_result)
        print(f"Vereinfacht: {simplified}")
        
        # Erwartetes Ergebnis
        expected = 2/(s+4)**2
        print(f"Erwartetes Ergebnis: {expected}")
        
        if sp.simplify(simplified - expected) == 0:
            print("✅ Manuelle Integration bestätigt: 2/(s+4)²")
        else:
            print("⚠️  Abweichung in manueller Integration")
            
    except Exception as e:
        print(f"❌ Integrationsfehler: {e}")


def analyze_sympy_internals():
    """Analysiere SymPy's interne Verarbeitung"""
    print("=" * 60)
    print("ANALYSE VON SYMPY'S INTERNER VERARBEITUNG")
    print("=" * 60)
    
    t, s = sp.symbols('t s', real=True, positive=True)
    f_t = 2*t*sp.exp(-4*t)
    
    print(f"Funktion: {f_t}")
    print(f"Typ: {type(f_t)}")
    print(f"Args: {f_t.args}")
    print(f"Faktoren: {f_t.as_ordered_factors()}")
    print()
    
    # Versuche verschiedene SymPy-Funktionen
    functions_to_test = [
        ("laplace_transform", lambda: sp.laplace_transform(f_t, t, s)),
        ("mellin_transform", lambda: sp.mellin_transform(f_t, t, s)),
        ("fourier_transform", lambda: sp.fourier_transform(f_t, t, s)),
    ]
    
    for name, func in functions_to_test:
        print(f"Test {name}:")
        try:
            result = func()
            print(f"  Ergebnis: {result}")
        except Exception as e:
            print(f"  Fehler: {e}")
        print()


if __name__ == "__main__":
    test_different_input_formats()
    print("\n")
    test_edge_cases()
    print("\n")
    test_manual_integration()
    print("\n")
    analyze_sympy_internals()
