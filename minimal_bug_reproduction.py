#!/usr/bin/env python3
"""
MINIMAL SYMPY BUG REPRODUCTION
==============================

Zeigt den SymPy Laplace-Transform Bug für String-Eingaben.
Das Problem: SymPy gibt für String-basierte Funktionen triviale f(t)/s Ergebnisse zurück.
"""

import sympy as sp

def test_different_conversion_methods():
    print("TESTING DIFFERENT STRING-TO-SYMPY CONVERSION METHODS")
    print("=" * 60)
    print(f"SymPy Version: {sp.__version__}")
    print()
    
    # Symbole
    t, s = sp.symbols('t s', real=True, positive=True)
    
    # Original string
    function_string = "2*t*exp(-4*t)"
    print(f"Original string: '{function_string}'")
    print()
    
    # Method 1: Standard sympify
    print("METHOD 1: Standard sp.sympify()")
    try:
        f1 = sp.sympify(function_string)
        print(f"  Result: {f1}")
        print(f"  Type: {type(f1)}")
        result1 = sp.laplace_transform(f1, t, s)
        F_s1 = result1[0] if isinstance(result1, tuple) else result1
        print(f"  Laplace: {F_s1}")
        if F_s1 == f1/s:
            print("  🐛 BUG: f(t)/s")
        else:
            print("  ✅ CORRECT")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print()
    
    # Method 2: sympify with evaluate=False
    print("METHOD 2: sp.sympify() with evaluate=False")
    try:
        f2 = sp.sympify(function_string, evaluate=False)
        print(f"  Result: {f2}")
        print(f"  Type: {type(f2)}")
        result2 = sp.laplace_transform(f2, t, s)
        F_s2 = result2[0] if isinstance(result2, tuple) else result2
        print(f"  Laplace: {F_s2}")
        if F_s2 == f2/s:
            print("  🐛 BUG: f(t)/s")
        else:
            print("  ✅ CORRECT")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print()
    
    # Method 3: Parse and reconstruct
    print("METHOD 3: Parse to dict and reconstruct")
    try:
        # Convert string to Python syntax first
        python_string = function_string.replace('exp', 'sp.exp')
        print(f"  Python syntax: {python_string}")
        
        # Create namespace for evaluation
        namespace = {'t': t, 's': s, 'sp': sp}
        f3 = eval(python_string, {'__builtins__': {}}, namespace)
        print(f"  Result: {f3}")
        print(f"  Type: {type(f3)}")
        result3 = sp.laplace_transform(f3, t, s)
        F_s3 = result3[0] if isinstance(result3, tuple) else result3
        print(f"  Laplace: {F_s3}")
        if F_s3 == f3/s:
            print("  🐛 BUG: f(t)/s")
        else:
            print("  ✅ CORRECT")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print()
    
    # Method 4: Step-by-step construction
    print("METHOD 4: Step-by-step manual construction")
    try:
        # Parse the string manually and construct step by step
        import re
        
        # Extract coefficient
        coeff_match = re.match(r'^(\d+)\*', function_string)
        coeff = int(coeff_match.group(1)) if coeff_match else 1
        
        # Extract exponential argument
        exp_match = re.search(r'exp\(([^)]+)\)', function_string)
        exp_arg = exp_match.group(1) if exp_match else None
        
        print(f"  Coefficient: {coeff}")
        print(f"  Exp argument: {exp_arg}")
        
        if exp_arg:
            # Convert exp argument
            exp_arg_sympy = sp.sympify(exp_arg)
            f4 = coeff * t * sp.exp(exp_arg_sympy)
        else:
            f4 = sp.sympify(function_string)
            
        print(f"  Result: {f4}")
        print(f"  Type: {type(f4)}")
        result4 = sp.laplace_transform(f4, t, s)
        F_s4 = result4[0] if isinstance(result4, tuple) else result4
        print(f"  Laplace: {F_s4}")
        if F_s4 == f4/s:
            print("  🐛 BUG: f(t)/s")
        else:
            print("  ✅ CORRECT")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print()
    
    # Method 5: Using lambdify and then back to symbolic
    print("METHOD 5: Via lambdify roundtrip")
    try:
        # First sympify normally
        f_temp = sp.sympify(function_string)
        
        # Convert to lambda function
        lambda_func = sp.lambdify(t, f_temp, 'sympy')
        
        # Apply to symbolic t to get back symbolic expression
        f5 = lambda_func(t)
        print(f"  Result: {f5}")
        print(f"  Type: {type(f5)}")
        result5 = sp.laplace_transform(f5, t, s)
        F_s5 = result5[0] if isinstance(result5, tuple) else result5
        print(f"  Laplace: {F_s5}")
        if F_s5 == f5/s:
            print("  🐛 BUG: f(t)/s")
        else:
            print("  ✅ CORRECT")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print()


def minimal_bug_demo():
    print("SYMPY LAPLACE-TRANSFORM BUG - MINIMAL REPRODUCTION")
    print("=" * 55)
    print(f"SymPy Version: {sp.__version__}")
    print()
    
    # Symbole
    t, s = sp.symbols('t s', real=True, positive=True)
    
    # Das GLEICHE mathematische Objekt, aber verschiedene Konstruktion:
    function_as_string = "2*t*exp(-4*t)"
    function_as_object = 2*t*sp.exp(-4*t)
    
    print("Test der GLEICHEN Funktion, aber verschiedene Konstruktion:")
    print(f"String-Version:  '{function_as_string}'")
    print(f"Objekt-Version:  {function_as_object}")
    print()
    
    # Parse String-Version
    f_from_string = sp.sympify(function_as_string)
    
    print("Nach dem Parsen sind beide identisch:")
    print(f"Aus String: {f_from_string}")
    print(f"Als Objekt:  {function_as_object}")
    print(f"Sind gleich: {f_from_string == function_as_object}")
    print()
    
    # Teste Laplace-Transform
    print("LAPLACE TRANSFORMATION:")
    print("-" * 25)
    
    # String-basierte Eingabe
    print("1. String-basierte Eingabe:")
    try:
        result_string = sp.laplace_transform(f_from_string, t, s)
        F_s_string = result_string[0] if isinstance(result_string, tuple) else result_string
        print(f"   Ergebnis: {F_s_string}")
        if F_s_string == f_from_string/s:
            print("   🐛 BUG: Triviales f(t)/s Ergebnis!")
        else:
            print("   ✅ Korrekt")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
    
    # Objekt-basierte Eingabe  
    print("2. Objekt-basierte Eingabe:")
    try:
        result_object = sp.laplace_transform(function_as_object, t, s)
        F_s_object = result_object[0] if isinstance(result_object, tuple) else result_object
        print(f"   Ergebnis: {F_s_object}")
        if F_s_object == function_as_object/s:
            print("   🐛 BUG: Triviales f(t)/s Ergebnis!")
        else:
            print("   ✅ Korrekt")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
    
    print()
    print("MATHEMATISCH KORREKT:")
    print("L{2*t*exp(-4*t)} = 2/(s+4)²")
    
    # Verifikation
    correct_result = 2/(s+4)**2
    try:
        inverse = sp.inverse_laplace_transform(correct_result, s, t)
        print(f"Verifikation: L⁻¹{{2/(s+4)²}} = {inverse}")
    except Exception as e:
        print(f"Verifikation nicht möglich: {e}")


if __name__ == "__main__":
    test_different_conversion_methods()
    print("\n" + "="*60 + "\n")
    minimal_bug_demo()
