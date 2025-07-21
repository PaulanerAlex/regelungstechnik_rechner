#!/usr/bin/env python3
"""
SymPy String Conversion Utility
===============================

Robuste String-zu-SymPy Konversion, die bekannte SymPy-Bugs umgeht,
insbesondere bei Laplace-        # Negative Lookahead für geschützte Namen und sp.
        protected_lookahead = r'(?!' + '|'.join(re.escape(p) for p in sorted_protected) + r')'
        if sorted_protected:  # Nur wenn es geschützte Namen gibt
            protected_lookahead = r'(?!' + '|'.join(re.escape(p) for p in sorted_protected) + r'|sp\.)'
        else:
            protected_lookahead = r'(?!sp\.)'ansformationen.

Unterstützt alle gängigen mathematischen Funktionen und Symbole.
"""

import sympy as sp
import re
from typing import Dict, Any, Optional

class SafeSymPyConverter:
    """
    Sichere String-zu-SymPy Konversion mit Bug-Workarounds
    """
    
    # Mapping von Standard-Funktionen zu SymPy-Funktionen
    FUNCTION_MAPPINGS = {
        'exp': 'sp.exp',
        'sin': 'sp.sin',
        'cos': 'sp.cos',
        'tan': 'sp.tan',
        'cot': 'sp.cot',
        'sec': 'sp.sec',
        'csc': 'sp.csc',
        'asin': 'sp.asin',
        'acos': 'sp.acos',
        'atan': 'sp.atan',
        'acot': 'sp.acot',
        'asec': 'sp.asec',
        'acsc': 'sp.acsc',
        'sinh': 'sp.sinh',
        'cosh': 'sp.cosh',
        'tanh': 'sp.tanh',
        'coth': 'sp.coth',
        'asinh': 'sp.asinh',
        'acosh': 'sp.acosh',
        'atanh': 'sp.atanh',
        'acoth': 'sp.acoth',
        'log': 'sp.log',
        'ln': 'sp.log',
        'log10': 'sp.log10',
        'sqrt': 'sp.sqrt',
        'abs': 'sp.Abs',
        'floor': 'sp.floor',
        'ceiling': 'sp.ceiling',
        'factorial': 'sp.factorial',
        'gamma': 'sp.gamma',
        'beta': 'sp.beta',
        'erf': 'sp.erf',
        'erfc': 'sp.erfc',
        'Heaviside': 'sp.Heaviside',
        'DiracDelta': 'sp.DiracDelta',
        'Max': 'sp.Max',
        'Min': 'sp.Min',
        'Piecewise': 'sp.Piecewise',
    }
    
    # Mathematische Konstanten
    CONSTANT_MAPPINGS = {
        'pi': 'sp.pi',
        'e': 'sp.E',
        'euler_gamma': 'sp.EulerGamma',
        'golden': 'sp.GoldenRatio',
        'oo': 'sp.oo',
        'inf': 'sp.oo',
        'infinity': 'sp.oo',
        'I': 'sp.I',
        'j': 'sp.I',
    }
    
    def __init__(self, default_symbols: Optional[Dict[str, sp.Symbol]] = None):
        """
        Args:
            default_symbols: Dictionary mit Standard-Symbolen wie {'t': t_symbol, 's': s_symbol}
        """
        if default_symbols is None:
            t, s = sp.symbols('t s', real=True, positive=True)
            self.default_symbols = {'t': t, 's': s}
        else:
            # Sicherheitsabfrage: Stelle sicher, dass es ein Dictionary ist
            if not isinstance(default_symbols, dict):
                raise TypeError(f"default_symbols muss ein Dictionary sein, erhalten: {type(default_symbols)}")
            self.default_symbols = default_symbols.copy()
    
    def _convert_to_python_syntax(self, expression_string: str) -> str:
        """
        Konvertiert mathematischen String zu Python/SymPy-Syntax
        
        Args:
            expression_string: Mathematischer Ausdruck als String
            
        Returns:
            Python-kompatible SymPy-Syntax
        """
        result = expression_string.strip()
        
        # 1. Spezielle Fälle zuerst
        # Potenzen: ^ sollte zu ** werden
        result = result.replace('^', '**')
        
        # 2. Funktionen ersetzen (längste zuerst, um Konflikte zu vermeiden)
        sorted_functions = sorted(self.FUNCTION_MAPPINGS.items(), 
                                key=lambda x: len(x[0]), reverse=True)
        
        for func, sp_func in sorted_functions:
            # Verwende Wort-Grenzen um Teilstring-Probleme zu vermeiden
            # Aber nur wenn es nicht Teil eines längeren Variablennamens ist
            pattern = r'\b' + re.escape(func) + r'(?=\s*\()'  # Nur wenn gefolgt von (
            result = re.sub(pattern, sp_func, result)
        
        # 3. Konstanten ersetzen (auch nur wenn alleinstehend)
        for const, sp_const in self.CONSTANT_MAPPINGS.items():
            pattern = r'\b' + re.escape(const) + r'\b(?!\w)'  # Nicht Teil eines längeren Worts
            result = re.sub(pattern, sp_const, result)
        
        # 4. Automatische Multiplikationserkennung hinzufügen (NACH Funktionserkennung!)
        result = self._add_implicit_multiplication(result)
        
        return result
    
    def _add_implicit_multiplication(self, expression: str) -> str:
        """
        Fügt automatisch Multiplikationszeichen (*) zwischen Ausdrücken hinzu.
        
        Erkennt Fälle wie:
        - 2s → 2*s
        - 3sp.sin(x) → 3*sp.sin(x)  
        - (s+1)(s+2) → (s+1)*(s+2)
        - 2(s+1) → 2*(s+1)
        - s(t-1) → s*(t-1)
        
        Args:
            expression: Mathematischer Ausdruck (nach Funktionskonversion)
            
        Returns:
            Ausdruck mit expliziten Multiplikationszeichen
        """
        result = expression.strip()
        
        # Geschützte Symbole (längere Namen die nicht getrennt werden sollen)
        protected_symbols = ['omega', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 
                           'eta', 'theta', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'rho', 
                           'sigma', 'tau', 'phi', 'chi', 'psi', 'Omega', 'Alpha', 'Beta',
                           'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Kappa',
                           'Lambda', 'Mu', 'Nu', 'Xi', 'Rho', 'Sigma', 'Tau', 'Phi', 
                           'Chi', 'Psi']
        
        # Sortiere nach Länge (längste zuerst) 
        sorted_protected = sorted(protected_symbols, key=len, reverse=True)
        
        # Pattern-basierte Ersetzungen (vorsichtige Reihenfolge)
        patterns = [
            # 1. Nummer vor Klammer: 2( → 2*(
            (r'(\d+)\s*(\()', r'\1*\2'),
            
            # 2. Klammer vor Klammer: )( → )*(  
            (r'(\))\s*(\()', r'\1*\2'),
            
            # 3. Klammer vor Symbol/Nummer: )s → )*s, )2 → )*2
            (r'(\))\s*([a-zA-Z_]\w*)', r'\1*\2'),
            (r'(\))\s*(\d+)', r'\1*\2'),
            
            # 4. Symbol vor Klammer: s( → s*(, aber NICHT sp.sin( → sp.*sin(
            (r'(?<!sp\.)([a-zA-Z_]\w*)\s*(\()', r'\1*\2'),
        ]
        
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        
        # 5. Nummer vor sp.Funktionen: 2sp.sin → 2*sp.sin
        # Erkenne sp.funktionsname und füge * davor ein wenn Nummer voransteht
        sp_function_pattern = r'(\d+)\s*(sp\.[a-zA-Z_]\w*)'
        result = re.sub(sp_function_pattern, r'\1*\2', result)
        
        # 6. Nummer vor geschützten Symbolen: 5omega → 5*omega
        for symbol in sorted_protected:
            pattern = r'(\d+)\s*(' + re.escape(symbol) + r')(?![a-zA-Z0-9_])'
            result = re.sub(pattern, r'\1*\2', result)
        
        # 7. Nummer vor einfachen Symbolen: 2s → 2*s
        # Negative Lookahead für geschützte Namen und sp.
        protected_lookahead = '(?!' + '|'.join(re.escape(p) for p in sorted_protected) + r'|sp\.)'
        simple_symbol_pattern = r'(\d+)\s*([a-zA-Z_]\w*)' + protected_lookahead
        
        def replace_simple_symbols(match):
            number = match.group(1)
            symbol = match.group(2)
            
            # Doppelcheck: Stelle sicher, dass es kein geschütztes Symbol ist
            if symbol in protected_symbols:
                return match.group(0)
                
            return f'{number}*{symbol}'
        
        result = re.sub(simple_symbol_pattern, replace_simple_symbols, result)
        
        # 8. Leerzeichen zwischen Symbolen: omega s → omega*s
        # Aber NICHT bei sp.funktionen oder Operatoren
        space_pattern = r'([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*)'
        
        def replace_spaces(match):
            sym1 = match.group(1)
            sym2 = match.group(2)
            
            # Nicht ersetzen bei sp.Funktionen oder Operatoren
            keywords = ['and', 'or', 'not', 'in', 'is', 'if', 'else', 'elif', 'for', 'while']
            if (sym1.startswith('sp') or sym2.startswith('sp') or 
                sym1 in keywords or sym2 in keywords):
                return match.group(0)
                
            return f'{sym1}*{sym2}'
        
        result = re.sub(space_pattern, replace_spaces, result)
        
        # 9. Einzelne Buchstaben nebeneinander: st → s*t
        # Aber NICHT wenn Teil von geschützten Namen oder sp.
        single_char_pattern = r'(?<!sp\.)(?<![a-zA-Z])([a-zA-Z])([a-zA-Z])(?![a-zA-Z])'
        
        def replace_single_chars(match):
            char1 = match.group(1)
            char2 = match.group(2)
            combined = char1 + char2
            
            # Prüfe ob Kombination geschützt ist
            if any(protected.startswith(combined) for protected in protected_symbols):
                return match.group(0)
                
            return f'{char1}*{char2}'
        
        result = re.sub(single_char_pattern, replace_single_chars, result)
        
        return result
    
    def _extract_symbols_from_string(self, expression_string: str) -> Dict[str, sp.Symbol]:
        """
        Extrahiert alle Symbole aus dem String und erstellt SymPy-Symbole
        """
        # Finde alle Variablen (Buchstaben, die keine Funktionen sind)
        # Berücksichtige griechische Buchstaben und Multi-Char-Symbole
        
        # Entferne bekannte Funktionen aus der Suche (nur wenn gefolgt von Klammern)
        temp_string = expression_string
        for func in self.FUNCTION_MAPPINGS.keys():
            temp_string = re.sub(r'\b' + re.escape(func) + r'(?=\s*\()', '', temp_string)
        
        # Entferne Konstanten (nur wenn alleinstehend)
        for const in self.CONSTANT_MAPPINGS.keys():
            temp_string = re.sub(r'\b' + re.escape(const) + r'\b(?!\w)', '', temp_string)
        
        # Finde Variable (Buchstaben, eventuell mit Zahlen/Unterstrichen)
        variables = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', temp_string)
        
        symbols = self.default_symbols.copy()
        
        # Erstelle Symbole für neue Variablen
        for var in set(variables):
            if var not in symbols and var != 'sp':  # 'sp' ist unser SymPy-Namespace
                # Versuche zu bestimmen, ob Symbol real/positiv sein sollte
                if var in ['t', 'tau', 'time']:
                    symbols[var] = sp.Symbol(var, real=True, positive=True)
                elif var in ['s', 'p', 'omega', 'w', 'freq']:
                    symbols[var] = sp.Symbol(var, real=True)
                elif var in ['n', 'm', 'k', 'i', 'j'] and var not in ['j']:  # j ist imaginäre Einheit
                    symbols[var] = sp.Symbol(var, integer=True)
                else:
                    symbols[var] = sp.Symbol(var, real=True)
        
        return symbols
    
    def safe_sympify(self, expression_string: str, 
                    additional_symbols: Optional[Dict[str, sp.Symbol]] = None,
                    test_laplace: bool = True) -> sp.Expr:
        """
        Sichere String-zu-SymPy Konversion mit Bug-Workarounds
        
        Args:
            expression_string: Mathematischer Ausdruck als String
            additional_symbols: Zusätzliche Symbole
            test_laplace: Ob Laplace-Transform-Bug-Test durchgeführt werden soll
            
        Returns:
            SymPy expression
        """
        if not expression_string or not expression_string.strip():
            raise ValueError("Leerer Ausdruck")
        
        # Symbole zusammenstellen
        symbols = self._extract_symbols_from_string(expression_string)
        if additional_symbols:
            symbols.update(additional_symbols)
        
        # METHOD 1: Python-Syntax Konversion
        try:
            python_string = self._convert_to_python_syntax(expression_string)
            
            # Sicherer eval mit begrenztem namespace
            namespace = symbols.copy()
            namespace['sp'] = sp
            
            result = eval(python_string, {'__builtins__': {}}, namespace)
            
            # Teste auf Laplace-Transform-Bug falls gewünscht
            if test_laplace and 't' in symbols and 's' in symbols:
                try:
                    test_result = sp.laplace_transform(result, symbols['t'], symbols['s'])
                    test_F_s = test_result[0] if isinstance(test_result, tuple) else test_result
                    
                    if test_F_s == result/symbols['s']:  # Bug erkannt
                        raise ValueError("Laplace-Transform Bug detected")
                except:
                    # Bei Fehlern im Laplace-Test ignorieren und Ergebnis verwenden
                    pass
            
            return result
            
        except Exception as e:
            # METHOD 2: Lambdify Roundtrip
            try:
                f_temp = sp.sympify(expression_string, locals=symbols)
                if 't' in symbols:
                    lambda_func = sp.lambdify(symbols['t'], f_temp, 'sympy')
                    result = lambda_func(symbols['t'])
                    
                    # Test auf Laplace-Bug
                    if test_laplace and 's' in symbols:
                        try:
                            test_result = sp.laplace_transform(result, symbols['t'], symbols['s'])
                            test_F_s = test_result[0] if isinstance(test_result, tuple) else test_result
                            if test_F_s != result/symbols['s']:
                                return result
                        except:
                            pass
                
                return f_temp
                
            except Exception as e2:
                # METHOD 3: Standard sympify als letzter Ausweg
                try:
                    return sp.sympify(expression_string, locals=symbols)
                except Exception as e3:
                    raise ValueError(f"Konnte Ausdruck nicht parsen: {expression_string}. "
                                   f"Fehler: {e}, {e2}, {e3}")


# Globale Instanz für einfache Verwendung
_default_converter = SafeSymPyConverter()

def safe_sympify(expression_string: str, 
                symbols: Optional[Dict[str, sp.Symbol]] = None,
                test_laplace: bool = True) -> sp.Expr:
    """
    Convenience-Funktion für sichere String-zu-SymPy Konversion
    
    Args:
        expression_string: Mathematischer Ausdruck als String
        symbols: Dictionary mit Symbolen
        test_laplace: Ob Laplace-Transform-Bug-Test durchgeführt werden soll
        
    Returns:
        SymPy expression
    """
    # Sicherheitsabfrage für symbols Parameter
    if symbols is not None and not isinstance(symbols, dict):
        error_msg = f"symbols muss ein Dictionary sein, erhalten: {type(symbols)}. Wert: {symbols}"
        raise TypeError(error_msg)
    
    if symbols:
        try:
            converter = SafeSymPyConverter(symbols)
            result = converter.safe_sympify(expression_string, test_laplace=test_laplace)
            return result
        except Exception as conv_safe_error:
            raise
    else:
        return _default_converter.safe_sympify(expression_string, test_laplace=test_laplace)


if __name__ == "__main__":
    # Umfassende Tests
    converter = SafeSymPyConverter()
    
    test_cases = [
        # Ursprüngliches Problem
        "2*t*exp(-4*t)",
        "t*exp(-2*t)",
        
        # Trigonometrische Funktionen
        "sin(2*t)",
        "cos(3*t)",
        "tan(t/2)",
        "sin(omega*t + phi)",
        
        # Hyperbolische Funktionen
        "sinh(a*t)",
        "cosh(t)",
        "tanh(2*t)",
        
        # Logarithmen
        "log(t)",
        "ln(t+1)",
        "log10(t)",
        
        # Komplexe Ausdrücke
        "exp(-a*t)*sin(omega*t + phi)",
        "t**2*exp(-sigma*t)*cos(omega*t)",
        "sqrt(t)*sin(2*pi*t)",
        
        # Konstanten
        "exp(-t)*sin(pi*t)",
        "e**(-t)*cos(2*pi*t)",
        
        # Spezielle Funktionen
        "Heaviside(t-1)",
        "DiracDelta(t)",
        "abs(t)",
        "sqrt(t**2 + 1)",
        
        # Potenzen mit ^
        "t^2*exp(-a*t)",
        "sin(omega*t)^2 + cos(omega*t)^2",
        
        # Griechische Buchstaben als Variablen
        "alpha*t*exp(-beta*t)",
        "gamma(n+1)/s^(n+1)",
    ]
    
    print("COMPREHENSIVE SAFE SYMPIFY TESTS")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i:2d}. Testing: {case}")
        print("-" * (len(case) + 12))
        
        try:
            result = converter.safe_sympify(case)
            print(f"    ✅ Success: {result}")
            print(f"    Type: {type(result)}")
            
            # Test Laplace if possible
            t, s = sp.symbols('t s', real=True, positive=True)
            if result.has(t):
                try:
                    laplace_result = sp.laplace_transform(result, t, s)
                    F_s = laplace_result[0] if isinstance(laplace_result, tuple) else laplace_result
                    if F_s != result/s:
                        print(f"    🎯 Laplace: {F_s}")
                    else:
                        print(f"    🐛 Laplace has bug: {F_s}")
                except Exception as le:
                    print(f"    ⚠️  Laplace error: {le}")
                    
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    
    print(f"\n{'='*50}")
    print("TEST COMPLETE")
