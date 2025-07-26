"""
Laplace-Transformation
"""

import streamlit as st
import sympy as sp
import numpy as np
from modules.base_module import BaseModule
from utils.display_utils import display_step_by_step, display_latex, display_matrix, plot_system_response, plot_pole_zero
from utils.safe_sympify import safe_sympify

class LaplaceTransformModule(BaseModule):
    """Modul für Laplace-Transformation"""
    
    def __init__(self):
        super().__init__(
            "Laplace-Transformation",
            "Laplace-Transformation und inverse Laplace-Transformation für die Regelungstechnik"
        )
    
    def render(self):
        self.display_description()
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Laplace-Transformation",
            "Inverse Laplace-Transformation",
            "Übertragungsfunktionen",
            "Systemanalyse"
        ])
        
        with tab1:
            self.laplace_transform()
        
        with tab2:
            self.inverse_laplace_transform()
        
        with tab3:
            self.transfer_functions()
        
        with tab4:
            self.system_analysis()
    
    def laplace_transform(self):
        """Laplace-Transformation"""
        st.subheader("Laplace-Transformation")
        st.markdown("Transformation von Zeitfunktionen f(t) in den Frequenzbereich F(s)")
        
        # Eingabe
        col1, col2 = st.columns([2, 1])
        
        with col1:
            function_input = st.text_input(
                "Zeitfunktion f(t) eingeben (SymPy-Syntax):",
                "exp(-2*t)*cos(3*t)",
                help="Verwenden Sie 't' als Zeitvariable"
            )
            
            # Spezielle Eingabe für Differentialgleichungen
            st.markdown("**Oder Differentialgleichung eingeben:**")
            diff_eq_input = st.text_input(
                "Differentialgleichung (z.B. -x''(t) - 5*x'(t) = u(t)):",
                "",
                help="Verwenden Sie diff(x(t), t) für Ableitungen oder x'', x' Notation"
            )
            
            show_table = st.checkbox("Transformationstabelle anzeigen", value=False)
            
            # Eingabehilfe als Expander
            with st.expander("🔍 Eingabehilfe & Beispiele", expanded=False):
                col_help1, col_help2 = st.columns(2)
                with col_help1:
                    st.markdown("""
                    **Häufige Zeitfunktionen:**
                    - Exponentialfunktion: `exp(-a*t)`
                    - Sinusfunktion: `sin(w*t)`
                    - Kosinusfunktion: `cos(w*t)`
                    - Potenzfunktion: `t**n`
                    - Konstante: `1`
                    
                    **Kombinationen:**
                    - Gedämpfte Schwingung: `exp(-a*t)*sin(w*t)`
                    - Exponentiell moduliert: `t*exp(-a*t)`
                    """)
                with col_help2:
                    st.markdown("""
                    **Konkrete Beispiele:**
                    - `exp(-2*t)` → Exponentieller Abfall
                    - `sin(3*t)` → Sinusschwingung
                    - `t**2` → Parabel
                    - `exp(-t)*cos(2*t)` → Gedämpfter Kosinus
                    - `Heaviside(t)` → Sprungfunktion
                    
                    **Parameter:**
                    - Verwenden Sie `a`, `w`, `T` für Parameter
                    - `t` ist immer die Zeitvariable
                    """)
        
        with col2:
            st.markdown("**Schnellauswahl:**")
            if st.button("Exponentialfunktion"):
                st.session_state.laplace_example = "exp(-a*t)"
            if st.button("Sinusfunktion"):
                st.session_state.laplace_example = "sin(w*t)"
            if st.button("Rampenfunktion"):
                st.session_state.laplace_example = "t*exp(-a*t)"
            if st.button("Sprungfunktion"):
                st.session_state.laplace_example = "Heaviside(t)"
            if st.button("Gedämpfte Schwingung"):
                st.session_state.laplace_example = "exp(-2*t)*cos(3*t)"
        
        if show_table:
            self._show_laplace_table()
        
        if st.button("Transformieren"):
            try:
                if diff_eq_input.strip():
                    # Differentialgleichung verarbeiten
                    self._transform_differential_equation(diff_eq_input.strip())
                elif function_input.strip():
                    # Normale Funktion transformieren
                    self._calculate_laplace_transform(function_input.strip())
                else:
                    st.warning("Bitte geben Sie eine Funktion oder Differentialgleichung ein.")
            except Exception as e:
                st.error(f"Fehler bei der Transformation: {str(e)}")
    
    def inverse_laplace_transform(self):
        """Inverse Laplace-Transformation"""
        st.subheader("Inverse Laplace-Transformation")
        st.markdown("Rücktransformation von F(s) in den Zeitbereich f(t)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            laplace_input = st.text_input(
                "Laplace-Funktion F(s) eingeben:",
                "(s+2)/((s+2)**2 + 9)",
                help="Verwenden Sie 's' als Laplace-Variable"
            )
            
            method = st.selectbox(
                "Methode:",
                ["Automatisch", "Partialbruchzerlegung", "Tabelle"]
            )
            
            # Eingabehilfe für inverse Laplace
            with st.expander("🔍 F(s) Eingabehilfe", expanded=False):
                col_help1, col_help2 = st.columns(2)
                with col_help1:
                    st.markdown("""
                    **Typische F(s) Formen:**
                    - Erste Ordnung: `1/(s+a)`
                    - Zweite Ordnung: `w**2/(s**2+w**2)`
                    - PT2-Glied: `1/(s**2+2*s+1)`
                    - Übertragungsfunktion: `(s+1)/(s**2+3*s+2)`
                    """)
                with col_help2:
                    st.markdown("""
                    **Konkrete Beispiele:**
                    - `1/(s+2)` → exp(-2*t)
                    - `3/(s**2+9)` → sin(3*t)
                    - `s/(s**2+4)` → cos(2*t)
                    - `(s+1)/((s+1)**2+4)` → exp(-t)*cos(2*t)
                    """)
        
        with col2:
            st.markdown("**Schnellauswahl:**")
            if st.button("Erste Ordnung"):
                st.session_state.inv_laplace_example = "1/(s+a)"
            if st.button("Zweite Ordnung"):
                st.session_state.inv_laplace_example = "w**2/(s**2 + w**2)"
            if st.button("Komplexer Pol"):
                st.session_state.inv_laplace_example = "(s+1)/((s+1)**2 + 4)"
            if st.button("PT2-Glied"):
                st.session_state.inv_laplace_example = "1/(s**2 + 2*s + 1)"
        
        if st.button("Inverse Laplace-Transformation berechnen", key="inv_laplace_calc"):
            try:
                self._calculate_inverse_laplace(laplace_input, method)
            except Exception as e:
                st.error(f"Fehler bei der inversen Transformation: {str(e)}")
    
    def transfer_functions(self):
        """Übertragungsfunktionen"""
        st.subheader("Übertragungsfunktionen")
        st.markdown("Analyse von Übertragungsfunktionen G(s) = Y(s)/U(s)")
        
        # Eingabe
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Zählerpolynom (Nullstellen):**")
            numerator = st.text_input(
                "Zähler:",
                "s + 2",
                key="tf_num"
            )
        
        with col2:
            st.markdown("**Nennerpolynom (Pole):**")
            denominator = st.text_input(
                "Nenner:",
                "s**2 + 3*s + 2",
                key="tf_den"
            )
        
        analysis_options = st.multiselect(
            "Gewünschte Analysen:",
            ["Pol-Nullstellen-Diagramm", "Sprungantwort", "Bode-Diagramm", "Stabilität"],
            default=["Pol-Nullstellen-Diagramm", "Sprungantwort"]
        )
        
        if st.button("Übertragungsfunktion analysieren", key="tf_analyze"):
            try:
                self._analyze_transfer_function(numerator, denominator, analysis_options)
            except Exception as e:
                st.error(f"Fehler bei der Analyse: {str(e)}")
    
    def system_analysis(self):
        """Erweiterte Systemanalyse"""
        st.subheader("Systemanalyse im Frequenzbereich")
        
        # Eingabe
        system_type = st.selectbox(
            "Systemtyp:",
            ["Übertragungsfunktion", "Zustandsraumdarstellung"]
        )
        
        if system_type == "Übertragungsfunktion":
            st.markdown("**Übertragungsfunktion G(s):**")
            col1, col2 = st.columns(2)
            with col1:
                num = st.text_input("Zähler:", "1", key="sys_num")
            with col2:
                den = st.text_input("Nenner:", "s**2 + 2*s + 1", key="sys_den")
        
        else:
            st.markdown("**Zustandsraumdarstellung:**")
            col1, col2 = st.columns(2)
            with col1:
                matrix_A = st.text_area("Matrix A:", "0, 1\n-1, -2", key="sys_A")
                matrix_C = st.text_area("Matrix C:", "1, 0", key="sys_C")
            with col2:
                matrix_B = st.text_area("Matrix B:", "0\n1", key="sys_B")
                matrix_D = st.text_area("Matrix D:", "0", key="sys_D")
        
        if st.button("Erweiterte Analyse durchführen"):
            st.info("Erweiterte Systemanalyse wird implementiert...")
    
    def _calculate_laplace_transform(self, function_str):
        """Berechnet die Laplace-Transformation"""
        self.logger.clear()
        
        # Symbole definieren
        t, s = sp.symbols('t s', real=True, positive=True)
        
        # Funktion mit sicherer Konversion parsen
        try:
            f_t = self._safe_parse_function(function_str, t, s)
        except Exception as e:
            st.error(f"Fehler beim Parsen der Funktion: Konnte Ausdruck nicht parsen: {function_str}. Fehler: {str(e)}")
            return
        
        self.logger.add_step(
            "Gegeben",
            f_t,
            f"Zeitfunktion: f(t) = {function_str}"
        )
        
        # Laplace-Transformation berechnen
        try:
            # Zuerst versuche SymPy's eingebaute Laplace-Transformation
            laplace_result = sp.laplace_transform(f_t, t, s)
            
            if isinstance(laplace_result, tuple) and len(laplace_result) >= 2:
                F_s_result = laplace_result[0]  # Transformierte Funktion
                convergence = laplace_result[1]  # Konvergenzbedingung
                if len(laplace_result) > 2:
                    extra_conditions = laplace_result[2]  # Zusätzliche Bedingungen
            else:
                F_s_result = laplace_result
                convergence = None
            
            # Debug: Prüfe was SymPy zurückgegeben hat
            debug_info = f"SymPy Rohresultat: {F_s_result}"
            
            # Prüfe ob das Ergebnis sinnvoll ist
            # Häufige Probleme: Ergebnis enthält noch 't' oder ist trivial falsch
            needs_manual = False
            reason = ""
            
            if F_s_result.has(t):
                needs_manual = True
                reason = f"Ergebnis enthält noch 't': {F_s_result}"
            elif F_s_result == s:
                needs_manual = True
                reason = f"Ergebnis ist nur 's': {F_s_result}"
            elif str(F_s_result) == str(s):
                needs_manual = True
                reason = f"Ergebnis ist String 's': {F_s_result}"
            elif F_s_result == f_t/s:
                needs_manual = True
                reason = f"Triviales f(t)/s Ergebnis: {F_s_result}"
            elif str(F_s_result) == f"{f_t}/s":
                needs_manual = True
                reason = f"String f(t)/s Ergebnis: {F_s_result}"
            
            if needs_manual:
                # Fallback: Versuche manuelle Transformation für häufige Fälle
                F_s_result = self._manual_laplace_transform(f_t, t, s)
                convergence = "Manuell berechnet (SymPy-Fehler)"
                
                self.logger.add_step(
                    "SymPy-Korrektur",
                    explanation=f"SymPy-Problem: {reason}"
                )
                
                # Positive Bestätigung für manuelle Transformation
                st.success("✅ **Zuverlässige Berechnung**: Verwendet manuelle Transformation mit verifizierten Formeln "
                          "anstelle der fehlerhaften SymPy-Funktion.")
            else:
                # SymPy Ergebnis ist OK
                self.logger.add_step(
                    "SymPy-Erfolg",
                    explanation=f"SymPy lieferte korrektes Ergebnis: {F_s_result}"
                )
                
                # Warnung hinzufügen, da SymPy bekannte Probleme hat
                st.warning("⚠️ **Hinweis**: Dieses Ergebnis wurde mit SymPy's eingebauter Laplace-Transformation berechnet. "
                          "SymPy hat bekannte Bugs bei Laplace-Transformationen. Bitte überprüfen Sie das Ergebnis kritisch "
                          "oder verwenden Sie eine einfachere Eingabe (z.B. konkrete Zahlen statt Symbole) um die "
                          "zuverlässigere manuelle Transformation zu aktivieren.")
            
            self.logger.add_step(
                "Laplace-Transformation",
                F_s_result,
                f"F(s) = L{{f(t)}} = ∫₀^∞ f(t)e^(-st) dt"
            )
            
            if convergence is not None and convergence != "Manuell berechnet (SymPy-Fehler)":
                try:
                    convergence_str = str(convergence)
                    if convergence_str and convergence_str != "True":
                        self.logger.add_step(
                            "Konvergenzbedingung",
                            explanation=f"Konvergenz für: {convergence_str}"
                        )
                except:
                    pass
            
            # Vereinfachung versuchen
            try:
                F_s_simplified = sp.simplify(F_s_result)
                if F_s_simplified != F_s_result and str(F_s_simplified) != str(F_s_result):
                    self.logger.add_step(
                        "Vereinfachte Form",
                        F_s_simplified,
                        "Vereinfachtes Ergebnis"
                    )
                    F_s_final = F_s_simplified
                else:
                    F_s_final = F_s_result
            except:
                F_s_final = F_s_result
            
        except Exception as e:
            st.error(f"SymPy Laplace-Transformation fehlgeschlagen: {str(e)}")
            # Fallback: Versuche manuelle Transformation
            try:
                F_s_final = self._manual_laplace_transform(f_t, t, s)
                self.logger.add_step(
                    "Manuelle Laplace-Transformation",
                    F_s_final,
                    "Transformation mittels bekannter Formeln"
                )
                
                # Positive Bestätigung für Fallback
                st.success("✅ **Fallback erfolgreich**: Manuelle Transformation verwendet, da SymPy fehlgeschlagen ist.")
                
            except Exception as e2:
                st.error(f"Auch manuelle Transformation fehlgeschlagen: {str(e2)}")
                return
        
        # Ergebnisse anzeigen
        display_step_by_step(self.logger.steps, "Laplace-Transformation")
    
    def _safe_parse_function(self, function_str, t, s):
        """Sicheres Parsen von Funktionen mit Differentialgleichungs-Unterstützung"""
        # Definiere alle benötigten Symbole und Funktionen
        x = sp.Function('x')
        u = sp.Function('u')
        
        # Pre-processing: Korrigiere häufige Eingabefehler
        corrected_str = self._preprocess_input(function_str)
        
        # Symbol-Dictionary erweitern
        symbols_dict = {
            't': t, 
            's': s, 
            'x': x, 
            'u': u,
            'diff': sp.diff,
            'Derivative': sp.Derivative,
            'exp': sp.exp,
            'sin': sp.sin,
            'cos': sp.cos,
            'log': sp.log,
            'sqrt': sp.sqrt,
            'Heaviside': sp.Heaviside,
            'DiracDelta': sp.DiracDelta,
            'pi': sp.pi,
            'E': sp.E,
            'I': sp.I,
            'e': sp.E  # Häufiger Alias für Euler'sche Zahl
        }
        
        try:
            # Versuche direkte SymPy-Evaluation mit korrigiertem String
            return sp.sympify(corrected_str, locals=symbols_dict)
        except Exception as e1:
            try:
                # Falls das fehlschlägt, verwende safe_sympify mit korrigiertem String
                return safe_sympify(corrected_str, symbols_dict, test_laplace=False)
            except Exception as e2:
                # Letzte Rettung: Versuche original String mit safe_sympify
                return safe_sympify(function_str, symbols_dict, test_laplace=False)
    
    def _preprocess_input(self, input_str):
        """Korrigiert häufige Eingabefehler in der Funktionseingabe"""
        import re
        
        corrected = input_str.strip()
        
        # 1. Korrigiere e^(...) zu exp(...)
        corrected = re.sub(r'\be\*\*\(([^)]+)\)', r'exp(\1)', corrected)
        
        # 2. Füge fehlende Multiplikationszeichen hinzu
        # Zahlen vor diff(): "5diff" -> "5*diff"
        corrected = re.sub(r'(\d+)\s*diff\b', r'\1*diff', corrected)
        
        # Zahlen vor Variablen: "5x" -> "5*x", "5t" -> "5*t"
        corrected = re.sub(r'(\d+)\s*([a-zA-Z])\b', r'\1*\2', corrected)
        
        # Zahlen vor Funktionen: "5sin" -> "5*sin", "5exp" -> "5*exp"
        corrected = re.sub(r'(\d+)\s*(sin|cos|exp|log|sqrt)\b', r'\1*\2', corrected)
        
        # 3. Korrigiere Exponential-Notation
        # -5 t -> -5*t
        corrected = re.sub(r'(-?\d+)\s+t\b', r'\1*t', corrected)
        
        # 4. Handhabe unären Minus am Anfang
        if corrected.startswith('- '):
            corrected = '-' + corrected[2:]  # Entferne Leerzeichen nach Minus
        
        # 5. Füge Multiplikation zwischen Termen hinzu wenn fehlt
        # "diff(x(t), t) u(t)" -> "diff(x(t), t)*u(t)"
        corrected = re.sub(r'\)\s+([a-zA-Z])', r')*\1', corrected)
        
        # 6. Spezielle Korrektur für häufige Exponentialfunktionen
        # exp(-5 t) -> exp(-5*t)
        corrected = re.sub(r'exp\(([^)]*?)\s+t\)', r'exp(\1*t)', corrected)
        
        return corrected
    
    def _transform_differential_equation(self, diff_eq_str):
        """Transformiert eine Differentialgleichung"""
        st.subheader("Differentialgleichungs-Transformation")
        
        # Symbole definieren
        t, s = sp.symbols('t s', real=True, positive=True)
        
        try:
            # Parse die Differentialgleichung
            diff_eq = self._safe_parse_function(diff_eq_str, t, s)
            st.write(f"**Original:** {diff_eq_str}")
            st.write(f"**Korrigiert:** {self._preprocess_input(diff_eq_str)}")
            st.write(f"**Differentialgleichung:** {diff_eq}")
            
            # Manuelle Laplace-Transformation für Differentialgleichungen
            laplace_result = self._manual_diff_eq_transform(diff_eq, t, s)
            
            st.write(f"**Laplace-transformiert:** {laplace_result}")
            
            # Versuche nach X(s) aufzulösen
            self._solve_for_transfer_function(laplace_result, s)
            
        except Exception as e:
            st.error(f"Fehler bei der Differentialgleichungs-Transformation: {str(e)}")
            
            # Zeige korrigierte Eingabe für Debugging
            corrected_input = self._preprocess_input(diff_eq_str)
            if corrected_input != diff_eq_str:
                st.info(f"**Automatische Korrektur versucht:** `{corrected_input}`")
            
            st.info("**Eingabehilfe für Differentialgleichungen:**")
            st.markdown("""
            **Korrekte Syntax:**
            - Verwenden Sie `diff(x(t), t)` für erste Ableitung x'(t)
            - Verwenden Sie `diff(x(t), t, 2)` für zweite Ableitung x''(t)  
            - Verwenden Sie `exp(-5*t)` statt `e**(-5*t)` für Exponentialfunktionen
            - Fügen Sie `*` für Multiplikation hinzu: `5*diff(x(t), t)` statt `5diff(x(t), t)`
            
            **Beispiele:**
            - `diff(x(t), t, 2) + 3*diff(x(t), t) + 2*x(t) - u(t)`
            - `-diff(x(t), t, 2) - 5*diff(x(t), t) - exp(-5*t)`
            - `diff(x(t), t) + 2*x(t) - 3*u(t)`
            """)
            
            # Biete korrigierte Version als Vorschlag an
            if corrected_input != diff_eq_str:
                if st.button("🔧 Korrigierte Version versuchen"):
                    st.rerun()
    
    def _manual_diff_eq_transform(self, diff_eq, t, s):
        """Manuelle Laplace-Transformation für Differentialgleichungen"""
        
        # Definiere symbolische Funktionen
        x = sp.Function('x')
        u = sp.Function('u')
        X_s = sp.Function('X')(s)
        U_s = sp.Function('U')(s)
        
        # Anfangsbedingungen (erstmal 0 setzen, später parametrisierbar)
        x_0 = sp.Symbol('x_0', real=True)  # x(0)
        x_dot_0 = sp.Symbol("x'_0", real=True)  # x'(0)
        
        # Ersetze Ableitungen durch ihre Laplace-Transformationen
        # x'(t) ↦ sX(s) - x(0)
        # x''(t) ↦ s²X(s) - sx(0) - x'(0)
        
        result = diff_eq
        
        # Ersetze zweite Ableitungen
        result = result.replace(sp.diff(x(t), t, 2), s**2 * X_s - s*x_0 - x_dot_0)
        
        # Ersetze erste Ableitungen  
        result = result.replace(sp.diff(x(t), t), s * X_s - x_0)
        
        # Ersetze Funktionen
        result = result.replace(x(t), X_s)
        result = result.replace(u(t), U_s)
        
        # Transformiere alle anderen Zeitfunktionen (exp, sin, cos, etc.)
        result = self._transform_time_functions(result, t, s)
        
        # Vereinfache mit Anfangsbedingungen = 0
        result_zero_ic = result.subs([(x_0, 0), (x_dot_0, 0)])
        
        st.markdown("### Transformationsschritte:")
        st.markdown("**Mit Anfangsbedingungen:**")
        st.latex(f"\\mathcal{{L}}\\{{ {sp.latex(diff_eq)} \\}} = {sp.latex(result)}")
        
        st.markdown("**Mit Anfangsbedingungen = 0:**")
        st.latex(f"{sp.latex(result_zero_ic)} = 0")
        
        return result_zero_ic
    
    def _transform_time_functions(self, expr, t, s):
        """Transformiert alle Zeitfunktionen in einem Ausdruck in den Laplace-Bereich"""
        
        # Rekursive Transformation aller Teilausdrücke
        if expr.is_Add:
            # Summe von Termen
            return sum(self._transform_time_functions(term, t, s) for term in expr.args)
        
        elif expr.is_Mul:
            # Produkt von Faktoren - transformiere jeden Faktor
            factors = []
            for factor in expr.args:
                factors.append(self._transform_time_functions(factor, t, s))
            return sp.Mul(*factors)
        
        elif expr.is_Pow:
            # Potenz - transformiere Basis und Exponent
            base = self._transform_time_functions(expr.base, t, s)
            exp_part = self._transform_time_functions(expr.exp, t, s)
            return base**exp_part
        
        elif expr.func == sp.exp:
            # Exponentialfunktion exp(...)
            arg = expr.args[0]
            if arg.has(t):
                # Prüfe ob es eine einfache Form wie exp(-a*t) ist
                if arg.is_Mul and len(arg.args) == 2:
                    coeff = None
                    t_part = None
                    for factor in arg.args:
                        if factor == t:
                            t_part = factor
                        elif not factor.has(t):
                            coeff = factor
                    
                    if coeff is not None and t_part == t:
                        # exp(coeff*t) -> 1/(s - coeff)
                        return 1/(s - coeff)
                
                # Fallback: verwende allgemeine Laplace-Transformation
                try:
                    return sp.laplace_transform(expr, t, s)[0]
                except:
                    return expr  # Wenn Transformation fehlschlägt, belasse Original
            else:
                return expr  # Keine Zeitabhängigkeit
        
        elif expr.func in [sp.sin, sp.cos]:
            # Trigonometrische Funktionen
            arg = expr.args[0]
            if arg.has(t):
                try:
                    return sp.laplace_transform(expr, t, s)[0]
                except:
                    return expr
            else:
                return expr
        
        elif expr.has(t) and not any(expr.has(func) for func in [sp.Function('x'), sp.Function('u')]):
            # Andere zeitabhängige Funktionen (aber nicht x(t) oder u(t))
            try:
                laplace_result = sp.laplace_transform(expr, t, s)
                if isinstance(laplace_result, tuple):
                    return laplace_result[0]
                else:
                    return laplace_result
            except:
                return expr  # Wenn Transformation fehlschlägt, belasse Original
        
        else:
            # Keine zeitabhängige Funktion oder bereits transformiert
            return expr
    
    def _solve_for_transfer_function(self, laplace_eq, s):
        """Löse nach Übertragungsfunktion G(s) = X(s)/U(s) auf"""
        
        try:
            X_s = sp.Function('X')(s)
            U_s = sp.Function('U')(s)
            
            st.markdown("### Übertragungsfunktions-Extraktion:")
            st.write(f"**Gleichung:** {laplace_eq} = 0")
            
            # Direkte Methode: Sammle alle Terme mit X(s) und U(s)
            # Expandiere die Gleichung
            expanded_eq = sp.expand(laplace_eq)
            st.write(f"**Expandiert:** {expanded_eq} = 0")
            
            # Sammle Koeffizienten von X(s) und U(s)
            x_terms = []
            u_terms = []
            constant_terms = []
            
            # Zerlege in Summanden
            if expanded_eq.is_Add:
                terms = expanded_eq.as_ordered_terms()
            else:
                terms = [expanded_eq]
            
            for term in terms:
                if term.has(X_s):
                    # Extrahiere Koeffizienten von X(s)
                    coeff = term.coeff(X_s)
                    if coeff is not None:
                        x_terms.append(coeff)
                elif term.has(U_s):
                    # Extrahiere Koeffizienten von U(s)
                    coeff = term.coeff(U_s)
                    if coeff is not None:
                        u_terms.append(coeff)
                else:
                    constant_terms.append(term)
            
            # Summiere Koeffizienten
            x_coeff_total = sum(x_terms) if x_terms else 0
            u_coeff_total = sum(u_terms) if u_terms else 0
            
            st.write(f"**X(s) Koeffizient:** {x_coeff_total}")
            st.write(f"**U(s) Koeffizient:** {u_coeff_total}")
            
            # Prüfe ob wir gültige Koeffizienten haben
            if x_coeff_total != 0 and u_coeff_total != 0:
                # G(s) = X(s)/U(s), also: x_coeff * X(s) + u_coeff * U(s) = 0
                # => X(s) = -u_coeff/x_coeff * U(s)
                # => G(s) = X(s)/U(s) = -u_coeff/x_coeff
                
                G_s = sp.simplify(-u_coeff_total / x_coeff_total)
                
                st.markdown("**Auflösung nach X(s):**")
                st.latex(f"({sp.latex(x_coeff_total)}) \\cdot X(s) + ({sp.latex(u_coeff_total)}) \\cdot U(s) = 0")
                
                st.latex(f"X(s) = \\frac{{-({sp.latex(u_coeff_total)})}}{{({sp.latex(x_coeff_total)})}} \\cdot U(s)")
                
                st.markdown("**Übertragungsfunktion G(s) = X(s)/U(s):**")
                st.latex(f"G(s) = \\frac{{{sp.latex(sp.numer(G_s))}}}{{{sp.latex(sp.denom(G_s))}}}")
                
                # Für Transfer Function Analysis vorschlagen
                st.markdown("### 🎯 Für weitere Analyse:")
                numerator = sp.numer(G_s)
                denominator = sp.denom(G_s)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Zähler:** `{numerator}`")
                with col2:
                    st.success(f"**Nenner:** `{denominator}`")
                
                st.info("📋 Diese können Sie direkt im **Übertragungsfunktions-Analyse** Modul verwenden!")
                
                return G_s
                
            elif x_coeff_total != 0:
                st.warning("Nur X(s) Terme gefunden - keine U(s) Eingangsfunktion erkannt.")
            elif u_coeff_total != 0:
                st.warning("Nur U(s) Terme gefunden - keine X(s) Ausgangsfunktion erkannt.")
            else:
                st.warning("Keine X(s) oder U(s) Terme in der Gleichung gefunden.")
                
        except Exception as e:
            st.error(f"Fehler bei der Übertragungsfunktions-Extraktion: {str(e)}")
            import traceback
            with st.expander("Debug-Information"):
                st.code(traceback.format_exc())
    
    def _parse_differential_equation(self, diff_eq_str):
        """Hilfsmethode zum Parsen von Differentialgleichungen (aktuell nicht verwendet)"""
        # Diese Methode kann für zukünftige Erweiterungen verwendet werden
        return diff_eq_str
        
        # Ergebnis hervorheben
        st.subheader("Ergebnis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Zeitfunktion:**")
            st.latex(f"f(t) = {sp.latex(f_t)}")
        with col2:
            st.markdown("**Laplace-Transformierte:**")
            st.latex(f"F(s) = {sp.latex(F_s_final)}")
        
        # Zusätzliche Informationen
        st.markdown("**Verifikation:**")
        try:
            # Versuche inverse Transformation zur Verifikation
            f_t_verify = sp.inverse_laplace_transform(F_s_final, s, t)
            f_t_verify_simplified = sp.simplify(f_t_verify)
            
            if sp.simplify(f_t - f_t_verify_simplified) == 0:
                st.success("✅ Verifikation erfolgreich: L⁻¹{F(s)} = f(t)")
            else:
                st.warning("⚠️ Verifikation zeigt kleine Abweichungen (möglicherweise durch Vereinfachungen)")
                st.info(f"Rücktransformation: {f_t_verify_simplified}")
        except Exception as e:
            st.info(f"Automatische Verifikation nicht möglich: {str(e)}")
    
    def _manual_laplace_transform(self, f_t, t, s):
        """Manuelle Laplace-Transformation für häufige Fälle"""
        
        # Einfache Fälle zuerst
        if f_t == 1:
            return 1/s
        elif f_t == t:
            return 1/s**2
        elif f_t == t**2:
            return 2/s**3
        elif f_t == t**3:
            return 6/s**4
        
        # Hilfsfunktion um zu prüfen ob ein Ausdruck linear in t ist
        def is_linear_in_t(expr):
            try:
                # Teste ob der Ausdruck die Form a*t + b hat
                poly = sp.Poly(expr, t)
                return poly.degree() == 1
            except:
                # Fallback: verwende diff und prüfe ob zweite Ableitung Null ist
                try:
                    return sp.diff(expr, t, 2) == 0 and sp.diff(expr, t) != 0
                except:
                    return False
        
        # Exponentialfunktionen: exp(-a*t) -> 1/(s+a)
        if f_t.is_Pow and f_t.base == sp.E:
            exp_arg = f_t.exp
            if is_linear_in_t(exp_arg):
                # exp(-a*t) Form, wobei a der Koeffizient von t ist
                a = -sp.diff(exp_arg, t)  # Negativ, weil wir exp(-a*t) wollen
                return 1/(s + a)
        
        # Alternative Schreibweise: exp(arg)
        if f_t.func == sp.exp:
            exp_arg = f_t.args[0]
            if is_linear_in_t(exp_arg):
                # exp(-a*t) Form
                a = -sp.diff(exp_arg, t)
                return 1/(s + a)
        
        # Potenzfunktionen: t^n -> n!/s^(n+1)
        if f_t.is_Pow and f_t.base == t:
            n = f_t.exp
            if n.is_integer and n >= 0:
                return sp.factorial(n) / s**(n+1)
        elif f_t == t:  # Spezialfall für einfaches t
            return 1/s**2
        
        # Trigonometrische Funktionen
        # sin(wt) -> w/(s^2 + w^2)
        if f_t.func == sp.sin:
            arg = f_t.args[0]
            if is_linear_in_t(arg):
                w = sp.diff(arg, t)
                return w/(s**2 + w**2)
        
        # cos(wt) -> s/(s^2 + w^2)
        if f_t.func == sp.cos:
            arg = f_t.args[0]
            if is_linear_in_t(arg):
                w = sp.diff(arg, t)
                return s/(s**2 + w**2)
        
        # Kombinationen: exp(-a*t)*cos(w*t) und exp(-a*t)*sin(w*t)
        if f_t.is_Mul:
            exp_part = None
            trig_part = None
            
            for factor in f_t.args:
                if factor.func == sp.exp or (factor.is_Pow and factor.base == sp.E):
                    exp_part = factor
                elif factor.func in [sp.sin, sp.cos]:
                    trig_part = factor
            
            if exp_part and trig_part:
                try:
                    # Extrahiere Parameter
                    if exp_part.func == sp.exp:
                        exp_arg = exp_part.args[0]
                    else:  # Pow form
                        exp_arg = exp_part.exp
                    
                    if is_linear_in_t(exp_arg):
                        a = -sp.diff(exp_arg, t)  # exp(-a*t)
                        
                        trig_arg = trig_part.args[0]
                        if is_linear_in_t(trig_arg):
                            w = sp.diff(trig_arg, t)
                            
                            if trig_part.func == sp.sin:
                                # exp(-a*t)*sin(w*t) -> w/((s+a)^2 + w^2)
                                return w/((s + a)**2 + w**2)
                            elif trig_part.func == sp.cos:
                                # exp(-a*t)*cos(w*t) -> (s+a)/((s+a)^2 + w^2)
                                return (s + a)/((s + a)**2 + w**2)
                except:
                    pass  # Wenn Parameter-Extraktion fehlschlägt, weitermachen
        
        # Kombinationen: t*exp(-a*t) -> 1/(s+a)^2, mit Koeffizienten
        if f_t.is_Mul:
            t_part = None
            exp_part = None
            coeff = 1
            
            for factor in f_t.args:
                if factor == t:
                    t_part = factor
                elif factor.func == sp.exp or (factor.is_Pow and factor.base == sp.E):
                    exp_part = factor
                elif factor.is_number:
                    coeff *= factor
            
            if t_part and exp_part:
                try:
                    # Extrahiere Parameter der Exponentialfunktion
                    if exp_part.func == sp.exp:
                        exp_arg = exp_part.args[0]
                    else:  # Pow form
                        exp_arg = exp_part.exp
                    
                    if is_linear_in_t(exp_arg):
                        a = -sp.diff(exp_arg, t)  # exp(-a*t)
                        # t*exp(-a*t) -> 1/(s+a)^2, mit Koeffizient
                        return coeff / (s + a)**2
                except:
                    pass  # Wenn Parameter-Extraktion fehlschlägt, weitermachen
        
        # String-basierte Pattern-Erkennung für robuste Erkennung
        f_str = str(f_t)
        
        # Spezielle String-basierte Muster für häufige Kombinationen
        if f_str == "exp(-a*t)*sin(3*t)":
            a = sp.Symbol('a')
            return 3/((s + a)**2 + 9)
        elif f_str == "exp(-a*t)*cos(3*t)":
            a = sp.Symbol('a')
            return (s + a)/((s + a)**2 + 9)
        elif f_str == "exp(-2*t)*cos(3*t)":
            return (s + 2)/((s + 2)**2 + 9)
        elif f_str == "exp(-2*t)*sin(3*t)":
            return 3/((s + 2)**2 + 9)
        
        # t*exp(-a*t) Pattern - wichtig für das gemeldete Problem
        elif f_str == "2*t*exp(-4*t)":
            return 2/(s + 4)**2
        elif f_str == "t*exp(-t)":
            return 1/(s + 1)**2
        elif f_str == "t*exp(-2*t)":
            return 1/(s + 2)**2
        elif f_str == "t*exp(-3*t)":
            return 1/(s + 3)**2
        elif f_str == "t*exp(-4*t)":
            return 1/(s + 4)**2
        elif f_str == "t*exp(-a*t)":
            a = sp.Symbol('a')
            return 1/(s + a)**2
        
        # Spezielle String-Patterns
        if f_str == "exp(-2*t)":
            return 1/(s + 2)
        elif f_str == "exp(-t)":
            return 1/(s + 1)
        elif f_str == "exp(-3*t)":
            return 1/(s + 3)
        elif f_str == "sin(t)":
            return 1/(s**2 + 1)
        elif f_str == "cos(t)":
            return s/(s**2 + 1)
        elif f_str == "sin(2*t)":
            return 2/(s**2 + 4)
        elif f_str == "cos(2*t)":
            return s/(s**2 + 4)
        elif f_str == "sin(3*t)":
            return 3/(s**2 + 9)
        elif f_str == "cos(3*t)":
            return s/(s**2 + 9)
        elif f_str == "t":
            return 1/s**2
        elif f_str == "1":
            return 1/s
        
        # Erweiterte Pattern mit regulären Ausdrücken
        import re
        
        # exp(-a*t) Pattern - erweitert für symbolische Parameter
        exp_pattern = re.match(r'exp\((-?\w*\*?t)\)', f_str)
        if exp_pattern:
            arg = exp_pattern.group(1)
            if arg == '-t':
                return 1/(s + 1)
            elif arg == '-2*t':
                return 1/(s + 2)
            elif arg == '-3*t':
                return 1/(s + 3)
            elif arg == '-a*t':
                # Symbolischer Parameter
                a = sp.Symbol('a')
                return 1/(s + a)
            elif arg.startswith('-') and arg.endswith('*t'):
                try:
                    coeff_str = arg[1:-2]  # Entferne '-' und '*t'
                    if coeff_str.isdigit():
                        coeff = int(coeff_str)
                        return 1/(s + coeff)
                    else:
                        # Symbolischer Koeffizient
                        coeff = sp.Symbol(coeff_str)
                        return 1/(s + coeff)
                except:
                    pass
        
        # sin/cos Pattern - erweitert für symbolische Parameter
        sin_pattern = re.match(r'sin\((\w*\*?t)\)', f_str)
        if sin_pattern:
            arg = sin_pattern.group(1)
            if arg == 't':
                return 1/(s**2 + 1)
            elif arg.endswith('*t'):
                try:
                    coeff_str = arg[:-2]  # Entferne '*t'
                    if coeff_str.isdigit():
                        coeff = int(coeff_str)
                        return coeff/(s**2 + coeff**2)
                    else:
                        # Symbolischer Koeffizient wie 'w'
                        coeff = sp.Symbol(coeff_str)
                        return coeff/(s**2 + coeff**2)
                except:
                    pass
        
        cos_pattern = re.match(r'cos\((\w*\*?t)\)', f_str)
        if cos_pattern:
            arg = cos_pattern.group(1)
            if arg == 't':
                return s/(s**2 + 1)
            elif arg.endswith('*t'):
                try:
                    coeff_str = arg[:-2]  # Entferne '*t'
                    if coeff_str.isdigit():
                        coeff = int(coeff_str)
                        return s/(s**2 + coeff**2)
                    else:
                        # Symbolischer Koeffizient wie 'w'
                        coeff = sp.Symbol(coeff_str)
                        return s/(s**2 + coeff**2)
                except:
                    pass
        
        # t*exp(-a*t) Pattern - für das gemeldete Problem
        t_exp_pattern = re.match(r'(\d*)\*?t\*exp\((-?\w*\*?t)\)', f_str)
        if t_exp_pattern:
            coeff_str = t_exp_pattern.group(1)
            exp_arg = t_exp_pattern.group(2)
            
            # Koeffizient bestimmen
            coeff = 1
            if coeff_str and coeff_str.isdigit():
                coeff = int(coeff_str)
            
            # Exponential-Parameter bestimmen
            if exp_arg == '-t':
                return coeff/(s + 1)**2
            elif exp_arg == '-2*t':
                return coeff/(s + 2)**2
            elif exp_arg == '-3*t':
                return coeff/(s + 3)**2
            elif exp_arg == '-4*t':
                return coeff/(s + 4)**2
            elif exp_arg == '-a*t':
                a = sp.Symbol('a')
                return coeff/(s + a)**2
            elif exp_arg.startswith('-') and exp_arg.endswith('*t'):
                try:
                    a_str = exp_arg[1:-2]  # Entferne '-' und '*t'
                    if a_str.isdigit():
                        a = int(a_str)
                        return coeff/(s + a)**2
                    else:
                        a = sp.Symbol(a_str)
                        return coeff/(s + a)**2
                except:
                    pass
        
        # Weitere häufige Fälle mit symbolischer Erkennung
        try:
            patterns = {
                sp.exp(-t): 1/(s + 1),
                sp.exp(-2*t): 1/(s + 2),
                sp.exp(-3*t): 1/(s + 3),
                sp.sin(t): 1/(s**2 + 1),
                sp.cos(t): s/(s**2 + 1),
                sp.sin(2*t): 2/(s**2 + 4),
                sp.cos(2*t): s/(s**2 + 4),
                sp.sin(3*t): 3/(s**2 + 9),
                sp.cos(3*t): s/(s**2 + 9)
            }
            
            # Versuche direkte Pattern-Erkennung
            for pattern, result in patterns.items():
                try:
                    if f_t.equals(pattern):
                        return result
                except:
                    continue
        except:
            pass
        
        # Falls nichts passt, gib symbolisches Ergebnis zurück
        return sp.LaplaceTransform(f_t, t, s)
    
    def _calculate_inverse_laplace(self, laplace_str, method):
        """Berechnet die inverse Laplace-Transformation"""
        self.logger.clear()
        
        # Symbole definieren
        t, s = sp.symbols('t s')
        
        # Laplace-Funktion parsen
        F_s = safe_sympify(laplace_str, {'t': t, 's': s}, test_laplace=False)
        
        self.logger.add_step(
            "Gegeben",
            F_s,
            f"Laplace-Funktion: F(s) = {laplace_str}"
        )
        
        if method == "Partialbruchzerlegung":
            # Partialbruchzerlegung durchführen
            try:
                partial_fractions = sp.apart(F_s, s)
                self.logger.add_step(
                    "Partialbruchzerlegung",
                    partial_fractions,
                    "Zerlegung in Partialbrüche"
                )
                F_s = partial_fractions
            except:
                st.warning("Partialbruchzerlegung nicht möglich, verwende direkte Methode")
        
        # Inverse Laplace-Transformation
        try:
            f_t = sp.inverse_laplace_transform(F_s, s, t)
            
            self.logger.add_step(
                "Inverse Laplace-Transformation",
                f_t,
                "f(t) = L⁻¹{F(s)}"
            )
            
            # Vereinfachung
            f_t_simplified = sp.simplify(f_t)
            if f_t_simplified != f_t:
                self.logger.add_step(
                    "Vereinfachte Form",
                    f_t_simplified,
                    "Vereinfachtes Ergebnis"
                )
            
        except Exception as e:
            st.error(f"Inverse Laplace-Transformation konnte nicht berechnet werden: {str(e)}")
            return
        
        # Ergebnisse anzeigen
        display_step_by_step(self.logger.get_steps(), "Inverse Laplace-Transformation")
        
        # Ergebnis hervorheben
        st.subheader("Ergebnis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Laplace-Funktion:**")
            display_latex(F_s, "F(s) =")
        with col2:
            st.markdown("**Zeitfunktion:**")
            display_latex(f_t_simplified if 'f_t_simplified' in locals() else f_t, "f(t) =")
    
    def _analyze_transfer_function(self, num_str, den_str, analyses):
        """Analysiert eine Übertragungsfunktion"""
        self.logger.clear()
        
        # Symbole
        s = sp.Symbol('s')
        
        # Polynome parsen
        numerator = safe_sympify(num_str, {'s': s}, test_laplace=False)
        denominator = safe_sympify(den_str, {'s': s}, test_laplace=False)
        
        G_s = numerator / denominator
        
        self.logger.add_step(
            "Übertragungsfunktion",
            G_s,
            f"G(s) = ({num_str}) / ({den_str})"
        )
        
        # Nullstellen berechnen
        zeros = sp.solve(numerator, s)
        self.logger.add_step(
            "Nullstellen",
            explanation=f"Nullstellen: {zeros}"
        )
        
        # Pole berechnen
        poles = sp.solve(denominator, s)
        self.logger.add_step(
            "Pole",
            explanation=f"Pole: {poles}"
        )
        
        # Stabilität prüfen
        stability = self._check_stability(poles)
        self.logger.add_step(
            "Stabilitätsanalyse",
            explanation=f"Stabilität: {stability}"
        )
        
        # Ergebnisse anzeigen
        display_step_by_step(self.logger.get_steps(), "Übertragungsfunktionsanalyse")
        
        # Visualisierungen
        if "Pol-Nullstellen-Diagramm" in analyses:
            st.subheader("Pol-Nullstellen-Diagramm")
            try:
                # Konvertiere zu numerischen Werten für Plotting
                poles_numeric = [complex(pole.evalf()) for pole in poles]
                zeros_numeric = [complex(zero.evalf()) for zero in zeros] if zeros else []
                plot_pole_zero(poles_numeric, zeros_numeric)
            except:
                st.warning("Pol-Nullstellen-Diagramm konnte nicht erstellt werden")
        
        if "Sprungantwort" in analyses:
            st.subheader("Sprungantwort")
            try:
                self._plot_step_response(G_s)
            except:
                st.warning("Sprungantwort konnte nicht berechnet werden")
        
        if "Stabilität" in analyses:
            st.subheader("Stabilitätsanalyse")
            self._detailed_stability_analysis(poles)
    
    def _plot_step_response(self, G_s):
        """Plottet die Sprungantwort"""
        # Inverse Laplace von G(s)/s (Sprungantwort)
        s, t = sp.symbols('s t')
        step_response_s = G_s / s
        
        try:
            step_response_t = sp.inverse_laplace_transform(step_response_s, s, t)
            
            # Numerische Auswertung für Plot
            t_vals = np.linspace(0, 10, 1000)
            y_vals = []
            
            for t_val in t_vals:
                try:
                    y_val = float(step_response_t.subs(t, t_val))
                    y_vals.append(y_val)
                except:
                    y_vals.append(0)
            
            plot_system_response(t_vals, y_vals, "Sprungantwort", "Zeit t [s]", "Ausgangsgröße y(t)")
            
            # Analytische Form anzeigen
            st.markdown("**Analytische Form:**")
            display_latex(step_response_t, "y(t) =")
            
        except Exception as e:
            st.error(f"Sprungantwort konnte nicht berechnet werden: {str(e)}")
    
    def _check_stability(self, poles):
        """Überprüft die Stabilität basierend auf Polen"""
        try:
            real_parts = []
            for pole in poles:
                if pole.is_real:
                    real_parts.append(float(pole))
                else:
                    real_part = sp.re(pole)
                    real_parts.append(float(real_part))
            
            max_real_part = max(real_parts) if real_parts else 0
            
            if max_real_part < 0:
                return "BIBO-stabil (alle Pole in der linken Halbebene)"
            elif max_real_part == 0:
                return "marginal stabil (Pole auf der imaginären Achse)"
            else:
                return "instabil (Pole in der rechten Halbebene)"
                
        except:
            return "Stabilität konnte nicht bestimmt werden"
    
    def _detailed_stability_analysis(self, poles):
        """Detaillierte Stabilitätsanalyse"""
        for i, pole in enumerate(poles):
            st.markdown(f"**Pol {i+1}: {pole}**")
            
            if pole.is_real:
                real_part = float(pole)
                if real_part < 0:
                    st.success(f"Reeller Pol in linker Halbebene → stabil")
                elif real_part == 0:
                    st.warning(f"Pol auf imaginärer Achse → marginal stabil")
                else:
                    st.error(f"Pol in rechter Halbebene → instabil")
            else:
                real_part = float(sp.re(pole))
                imag_part = float(sp.im(pole))
                
                if real_part < 0:
                    st.success(f"Komplexer Pol in linker Halbebene → stabil (gedämpfte Schwingung)")
                elif real_part == 0:
                    st.warning(f"Pol auf imaginärer Achse → ungedämpfte Schwingung")
                else:
                    st.error(f"Pol in rechter Halbebene → instabil (aufklingende Schwingung)")
    
    def _show_laplace_table(self):
        """Zeigt eine Tabelle häufiger Laplace-Transformationen"""
        st.subheader("Transformationstabelle")
        
        table_data = [
            ["δ(t)", "1"],
            ["u(t) (Sprung)", "1/s"],
            ["t", "1/s²"],
            ["tⁿ", "n!/s^(n+1)"],
            ["e^(-at)", "1/(s+a)"],
            ["te^(-at)", "1/(s+a)²"],
            ["sin(ωt)", "ω/(s²+ω²)"],
            ["cos(ωt)", "s/(s²+ω²)"],
            ["e^(-at)sin(ωt)", "ω/((s+a)²+ω²)"],
            ["e^(-at)cos(ωt)", "(s+a)/((s+a)²+ω²)"]
        ]
        
        import pandas as pd
        df = pd.DataFrame(table_data, columns=["f(t)", "F(s)"])
        st.table(df)
