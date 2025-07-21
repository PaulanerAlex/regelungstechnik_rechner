#!/usr/bin/env python3
"""
Workaround für SymPy Laplace-Transform Bug
Implementiert alternative String-zu-SymPy Konversion
"""

import sympy as sp
import re

def safe_sympify_for_laplace(function_string, symbols_dict=None):
    """
    Sichere String-zu-SymPy Konversion, die den Laplace-Transform Bug umgeht
    
    Args:
        function_string: String mit mathematischer Funktion
        symbols_dict: Dictionary mit Symbolen {name: symbol}
    
    Returns:
        SymPy expression die korrekt mit laplace_transform() funktioniert
    """
    if symbols_dict is None:
        t, s = sp.symbols('t s', real=True, positive=True)
        symbols_dict = {'t': t, 's': s}
    
    print(f"Original string: '{function_string}'")
    
    # METHOD 1: Versuche Python-Syntax Konversion
    try:
        # Ersetze mathematische Funktionen mit sp. Präfix
        python_string = function_string.replace('exp', 'sp.exp')
        python_string = python_string.replace('sin', 'sp.sin')
        python_string = python_string.replace('cos', 'sp.cos')
        python_string = python_string.replace('log', 'sp.log')
        python_string = python_string.replace('sqrt', 'sp.sqrt')
        
        print(f"Python syntax: '{python_string}'")
        
        # Sicherer eval mit begrenztem namespace
        namespace = symbols_dict.copy()
        namespace['sp'] = sp
        
        result = eval(python_string, {'__builtins__': {}}, namespace)
        print(f"Method 1 result: {result}")
        
        # Teste ob Laplace-Transform funktioniert
        t, s = symbols_dict.get('t'), symbols_dict.get('s')
        if t and s:
            test_result = sp.laplace_transform(result, t, s)
            test_F_s = test_result[0] if isinstance(test_result, tuple) else test_result
            
            if test_F_s != result/s:  # Nicht der triviale f(t)/s Bug
                print("✅ Method 1 successful - no bug detected")
                return result
            else:
                print("🐛 Method 1 has bug - trying fallback")
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # FALLBACK: Lambdify roundtrip
    try:
        print("Trying lambdify roundtrip...")
        f_temp = sp.sympify(function_string)
        t = symbols_dict.get('t', sp.Symbol('t'))
        
        lambda_func = sp.lambdify(t, f_temp, 'sympy')
        result = lambda_func(t)
        print(f"Lambdify result: {result}")
        
        # Teste Laplace-Transform
        s = symbols_dict.get('s', sp.Symbol('s'))
        test_result = sp.laplace_transform(result, t, s)
        test_F_s = test_result[0] if isinstance(test_result, tuple) else test_result
        
        if test_F_s != result/s:
            print("✅ Lambdify method successful")
            return result
        else:
            print("🐛 Lambdify method also has bug")
            
    except Exception as e:
        print(f"Lambdify failed: {e}")
    
    # LAST RESORT: Standard sympify mit Warnung
    print("⚠️  Using standard sympify - may have bugs")
    return sp.sympify(function_string)


def test_workaround():
    """Teste den Workaround"""
    print("TESTING SYMPY LAPLACE BUG WORKAROUND")
    print("=" * 45)
    
    t, s = sp.symbols('t s', real=True, positive=True)
    symbols = {'t': t, 's': s}
    
    test_cases = [
        "2*t*exp(-4*t)",
        "t*exp(-2*t)",
        "3*exp(-t)",
        "sin(2*t)",
        "cos(3*t)*exp(-t)",
        "t**2*exp(-a*t)"
    ]
    
    for case in test_cases:
        print(f"\n--- Testing: {case} ---")
        
        # Standard sympify
        f_standard = sp.sympify(case)
        try:
            result_standard = sp.laplace_transform(f_standard, t, s)
            F_s_standard = result_standard[0] if isinstance(result_standard, tuple) else result_standard
            print(f"Standard: {F_s_standard}")
            has_bug_standard = (F_s_standard == f_standard/s)
        except:
            print("Standard: ERROR")
            has_bug_standard = True
        
        # Workaround
        try:
            f_workaround = safe_sympify_for_laplace(case, symbols)
            result_workaround = sp.laplace_transform(f_workaround, t, s)
            F_s_workaround = result_workaround[0] if isinstance(result_workaround, tuple) else result_workaround
            print(f"Workaround: {F_s_workaround}")
            has_bug_workaround = (F_s_workaround == f_workaround/s)
        except Exception as e:
            print(f"Workaround: ERROR - {e}")
            has_bug_workaround = True
        
        # Vergleich
        if has_bug_standard and not has_bug_workaround:
            print("🎉 WORKAROUND FIXES THE BUG!")
        elif not has_bug_standard and not has_bug_workaround:
            print("✅ Both methods work")
        elif has_bug_standard and has_bug_workaround:
            print("❌ Both methods have bugs")
        else:
            print("⚠️  Mixed results")


if __name__ == "__main__":
    test_workaround()
