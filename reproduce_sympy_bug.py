#!/usr/bin/env python3
"""
Minimal reproduction of SymPy Laplace transform bug
Zeigt den Bug in SymPy's laplace_transform() Funktion
"""

import sympy as sp

def reproduce_sympy_bug():
    """Reproduziert den SymPy Laplace-Transform Bug minimal"""
    print("=" * 60)
    print("SYMPY LAPLACE-TRANSFORM BUG REPRODUCTION")
    print("=" * 60)
    
    # Symbole definieren
    t, s = sp.symbols('t s', real=True, positive=True)
    
    # Test-Funktionen, die den Bug zeigen
    test_functions = [
        "2*t*exp(-4*t)",     # Das ursprünglich gemeldete Problem
        "t*exp(-2*t)",       # Einfacherer Fall
        "3*t*exp(-t)",       # Noch ein Fall
        "t*exp(-a*t)"        # Symbolischer Fall
    ]
    
    print(f"SymPy Version: {sp.__version__}")
    print()
    
    for func_str in test_functions:
        print(f"Test: {func_str}")
        print("-" * 40)
        
        # Parse die Funktion
        f_t = sp.sympify(func_str)
        print(f"Parsed: {f_t}")
        
        # Erwartetes Ergebnis (mathematisch korrekt)
        if func_str == "2*t*exp(-4*t)":
            expected = "2/(s+4)**2"
        elif func_str == "t*exp(-2*t)":
            expected = "1/(s+2)**2"
        elif func_str == "3*t*exp(-t)":
            expected = "3/(s+1)**2"
        elif func_str == "t*exp(-a*t)":
            expected = "1/(s+a)**2"
        
        print(f"Mathematisch korrekt (L{{t*exp(-a*t)}} = 1/(s+a)²): {expected}")
        
        # SymPy's Laplace-Transform
        try:
            result = sp.laplace_transform(f_t, t, s)
            
            if isinstance(result, tuple):
                F_s = result[0]
                convergence = result[1] if len(result) > 1 else "unknown"
                print(f"SymPy Ergebnis: {F_s}")
                print(f"Konvergenz: {convergence}")
                
                # Prüfe auf den spezifischen Bug
                if F_s == f_t/s:
                    print("🐛 BUG DETECTED: SymPy gibt triviales f(t)/s zurück!")
                elif str(F_s) == f"{f_t}/s":
                    print("🐛 BUG DETECTED: SymPy gibt String f(t)/s zurück!")
                elif F_s.has(t):
                    print("🐛 BUG DETECTED: Ergebnis enthält noch die Zeitvariable 't'!")
                else:
                    print("✅ SymPy Ergebnis scheint korrekt")
                    
                    # Vergleiche mit erwartetem Ergebnis
                    expected_expr = sp.sympify(expected.replace('a', str(sp.Symbol('a'))))
                    if sp.simplify(F_s - expected_expr) == 0:
                        print("✅ Ergebnis ist mathematisch korrekt!")
                    else:
                        print("⚠️  Ergebnis weicht vom erwarteten ab")
                        
            else:
                print(f"SymPy Ergebnis (nicht-tuple): {result}")
                
        except Exception as e:
            print(f"❌ SymPy Fehler: {e}")
        
        print()


def demonstrate_correct_calculation():
    """Zeigt die korrekte manuelle Berechnung"""
    print("=" * 60)
    print("KORREKTE MANUELLE BERECHNUNG")
    print("=" * 60)
    
    t, s, a = sp.symbols('t s a', real=True, positive=True)
    
    print("Laplace-Transform Regel für t*exp(-a*t):")
    print("L{t*exp(-a*t)} = 1/(s+a)²")
    print()
    
    print("Anwendung auf konkrete Fälle:")
    cases = [
        ("t*exp(-t)", "a=1", "1/(s+1)²"),
        ("t*exp(-2*t)", "a=2", "1/(s+2)²"), 
        ("2*t*exp(-4*t)", "a=4, Koeff=2", "2/(s+4)²"),
        ("3*t*exp(-t)", "a=1, Koeff=3", "3/(s+1)²")
    ]
    
    for func, params, result in cases:
        print(f"{func} -> {params} -> {result}")
    
    print()
    print("Verifikation durch inverse Transformation:")
    
    # Beispiel: L⁻¹{2/(s+4)²} = 2*t*exp(-4*t)
    F_s = 2/(s+4)**2
    try:
        f_t_verify = sp.inverse_laplace_transform(F_s, s, t)
        print(f"L⁻¹{{2/(s+4)²}} = {f_t_verify}")
        
        original = 2*t*sp.exp(-4*t)
        if sp.simplify(f_t_verify - original) == 0:
            print("✅ Inverse Transformation bestätigt die Korrektheit!")
        else:
            print(f"Original: {original}")
            print(f"Differenz: {sp.simplify(f_t_verify - original)}")
            
    except Exception as e:
        print(f"Inverse Transformation Fehler: {e}")


if __name__ == "__main__":
    reproduce_sympy_bug()
    print("\n")
    demonstrate_correct_calculation()
