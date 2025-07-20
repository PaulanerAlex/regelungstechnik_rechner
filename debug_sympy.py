#!/usr/bin/env python3
"""
Detaillierter Test für SymPy Laplace-Transformation Problem
"""

import sympy as sp

def debug_sympy_issue():
    """Versuche das SymPy-Problem zu reproduzieren"""
    print("=== Debug SymPy Laplace-Transformation ===")
    
    # Exakt gleiche Symbole wie im Code
    t, s = sp.symbols('t s', real=True, positive=True)
    
    test_cases = [
        "exp(-2*t)",
        "exp(-a*t)", 
        "sin(3*t)",
        "t"
    ]
    
    for case in test_cases:
        print(f"\n--- Test Case: {case} ---")
        
        # Parse wie im echten Code
        f_t = sp.sympify(case)
        print(f"Parsed function: {f_t}")
        print(f"Function type: {type(f_t)}")
        
        # Laplace-Transform
        result = sp.laplace_transform(f_t, t, s)
        print(f"Raw result: {result}")
        print(f"Result type: {type(result)}")
        
        # Tuple handling wie im Code
        if isinstance(result, tuple) and len(result) >= 2:
            F_s = result[0]
            convergence = result[1]
            print(f"F(s): {F_s}")
            print(f"Convergence: {convergence}")
        else:
            F_s = result
            print(f"Direct F(s): {F_s}")
        
        # Tests wie im Code
        print(f"F_s.has(t): {F_s.has(t)}")
        print(f"F_s == s: {F_s == s}")
        print(f"str(F_s) == str(s): {str(F_s) == str(s)}")
        print(f"F_s == f_t/s: {F_s == f_t/s}")
        print(f"f_t/s = {f_t/s}")
        print(f"str(F_s) == f'{f_t}/s': {str(F_s) == f'{f_t}/s'}")
        
        # Welche Bedingung würde triggern?
        needs_manual = (F_s.has(t) or 
                       F_s == s or 
                       str(F_s) == str(s) or
                       F_s == f_t/s or
                       str(F_s) == f"{f_t}/s")
        
        print(f"Would trigger manual: {needs_manual}")

if __name__ == "__main__":
    debug_sympy_issue()
