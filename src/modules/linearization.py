"""
Linearisierung von Differentialgleichungen
"""

import streamlit as st
import sympy as sp
from modules.base_module import BaseModule
from utils.display_utils import display_step_by_step, display_latex, display_matrix

class LinearizationModule(BaseModule):
    """Modul für Linearisierung von Differentialgleichungen"""
    
    def __init__(self):
        super().__init__(
            "Linearisierung",
            "Linearisierung nichtlinearer Differentialgleichungen um Arbeitspunkte"
        )
    
    def render(self):
        self.display_description()
        
        tab1, tab2 = st.tabs([
            "Autonome Systeme",
            "Systeme mit Eingang"
        ])
        
        with tab1:
            self.linearize_autonomous_system()
        
        with tab2:
            self.linearize_system_with_input()
    
    def linearize_autonomous_system(self):
        """Linearisierung autonomer Systeme"""
        st.subheader("Linearisierung autonomer Systeme")
        st.markdown("Für Systeme der Form: $\\dot{\\mathbf{x}} = \\mathbf{f}(\\mathbf{x})$")
        
        # Eingabe
        st.markdown("**Nichtlineares System eingeben:**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            system_input = st.text_area(
                "Systemgleichungen (eine pro Zeile, SymPy-Syntax):",
                "x2\n-sin(x1) - 0.1*x2",
                height=100,
                help="Verwenden Sie x1, x2, x3, ... für die Zustandsvariablen"
            )
            
            equilibrium_input = st.text_input(
                "Gleichgewichtspunkt (x1, x2, ...):",
                "0, 0",
                help="Komma-getrennte Werte"
            )
        
        with col2:
            st.markdown("**Beispiele:**")
            if st.button("Pendel"):
                st.session_state.lin_example = "pendulum"
            if st.button("Van der Pol"):
                st.session_state.lin_example = "vanderpol"
            if st.button("Duffing"):
                st.session_state.lin_example = "duffing"
        
        if st.button("Linearisierung berechnen", key="lin_auto"):
            try:
                self._linearize_autonomous(system_input, equilibrium_input)
            except Exception as e:
                st.error(f"Fehler bei der Linearisierung: {str(e)}")
    
    def linearize_system_with_input(self):
        """Linearisierung von Systemen mit Eingang"""
        st.subheader("Linearisierung von Systemen mit Eingang")
        st.markdown("Für Systeme der Form: $\\dot{\\mathbf{x}} = \\mathbf{f}(\\mathbf{x}, \\mathbf{u})$")
        
        # Eingabe
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Systemgleichungen:**")
            system_input = st.text_area(
                "f(x,u) (eine Gleichung pro Zeile):",
                "x2\n-sin(x1) - 0.1*x2 + u1",
                height=100,
                key="lin_input_sys"
            )
            
            equilibrium_x = st.text_input(
                "Gleichgewichtspunkt x*:",
                "0, 0",
                key="lin_input_x"
            )
        
        with col2:
            st.markdown("**Eingangsvariablen:**")
            input_vars = st.text_input(
                "Eingangsvariablen (u1, u2, ...):",
                "u1",
                help="Komma-getrennte Variable"
            )
            
            equilibrium_u = st.text_input(
                "Gleichgewichtseingänge u*:",
                "0",
                key="lin_input_u"
            )
        
        if st.button("Linearisierung mit Eingang berechnen", key="lin_input"):
            try:
                self._linearize_with_input(system_input, input_vars, equilibrium_x, equilibrium_u)
            except Exception as e:
                st.error(f"Fehler bei der Linearisierung: {str(e)}")
    
    def _linearize_autonomous(self, system_str, equilibrium_str):
        """Linearisiert ein autonomes System"""
        self.logger.clear()
        
        # System parsen
        equations = system_str.strip().split('\n')
        n = len(equations)
        
        # Zustandsvariablen definieren
        x_vars = [sp.Symbol(f'x{i+1}') for i in range(n)]
        
        # Systemfunktionen parsen
        f_funcs = []
        for eq in equations:
            f_func = sp.sympify(eq)
            f_funcs.append(f_func)
        
        # Gleichgewichtspunkt parsen
        eq_values = [float(x.strip()) for x in equilibrium_str.split(',')]
        equilibrium_point = dict(zip(x_vars, eq_values))
        
        self.logger.add_step(
            "Nichtlineares System",
            explanation=f"Gegeben: ẋ = f(x) mit f(x) = [{', '.join(str(f) for f in f_funcs)}]"
        )
        
        # Gleichgewichtspunkt überprüfen
        eq_check = [f.subs(equilibrium_point) for f in f_funcs]
        
        self.logger.add_step(
            "Gleichgewichtspunkt-Verifikation",
            explanation=f"Überprüfe f({eq_values}) = 0: {eq_check}"
        )
        
        all_zero = all(abs(float(val)) < 1e-10 for val in eq_check)
        if not all_zero:
            st.warning("Achtung: Der angegebene Punkt ist möglicherweise kein exakter Gleichgewichtspunkt!")
        
        # Jacobi-Matrix berechnen
        self.logger.add_step(
            "Jacobi-Matrix berechnen",
            explanation="Berechne A = ∂f/∂x am Gleichgewichtspunkt"
        )
        
        A_symbolic = sp.Matrix([[sp.diff(f, x) for x in x_vars] for f in f_funcs])
        
        self.logger.add_step(
            "Symbolische Jacobi-Matrix",
            A_symbolic,
            "A = ∂f/∂x"
        )
        
        # Am Gleichgewichtspunkt auswerten
        A_linearized = A_symbolic.subs(equilibrium_point)
        
        self.logger.add_step(
            "Linearisierte Systemmatrix",
            A_linearized,
            f"A = ∂f/∂x|_(x*) mit x* = {eq_values}"
        )
        
        # Eigenwerte der linearisierten Matrix
        eigenvals = A_linearized.eigenvals()
        
        self.logger.add_step(
            "Eigenwerte des linearisierten Systems",
            explanation=f"Eigenwerte: {list(eigenvals.keys())}"
        )
        
        # Stabilitätsanalyse
        stability = self._analyze_stability(eigenvals)
        
        self.logger.add_step(
            "Stabilitätsanalyse",
            explanation=f"Stabilität: {stability}"
        )
        
        # Ergebnisse anzeigen
        display_step_by_step(self.logger.get_steps(), "Linearisierung des autonomen Systems")
        
        # Zusammenfassung
        st.subheader("Linearisiertes System")
        st.latex(r"\\dot{\\mathbf{\\xi}} = \\mathbf{A} \\mathbf{\\xi}")
        st.markdown("mit $\\mathbf{\\xi} = \\mathbf{x} - \\mathbf{x}^*$ (Abweichung vom Gleichgewichtspunkt)")
        
        display_matrix(A_linearized, "Linearisierte Systemmatrix A")
        
        # Eigenwerte und Stabilität
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Eigenwerte:**")
            for eigenval in eigenvals.keys():
                st.latex(f"\\lambda = {sp.latex(eigenval)}")
        
        with col2:
            st.markdown("**Stabilität:**")
            if stability == "asymptotisch stabil":
                st.success(f"Das System ist {stability}")
            elif stability == "instabil":
                st.error(f"Das System ist {stability}")
            else:
                st.warning(f"Das System ist {stability}")
    
    def _linearize_with_input(self, system_str, input_vars_str, equilibrium_x_str, equilibrium_u_str):
        """Linearisiert ein System mit Eingang"""
        self.logger.clear()
        
        # System parsen
        equations = system_str.strip().split('\n')
        n = len(equations)
        
        # Variablen definieren
        x_vars = [sp.Symbol(f'x{i+1}') for i in range(n)]
        u_vars = [sp.Symbol(var.strip()) for var in input_vars_str.split(',')]
        
        # Systemfunktionen parsen
        f_funcs = []
        for eq in equations:
            f_func = sp.sympify(eq)
            f_funcs.append(f_func)
        
        # Gleichgewichtspunkte parsen
        eq_x_values = [float(x.strip()) for x in equilibrium_x_str.split(',')]
        eq_u_values = [float(u.strip()) for u in equilibrium_u_str.split(',')]
        
        equilibrium_point = dict(zip(x_vars + u_vars, eq_x_values + eq_u_values))
        
        self.logger.add_step(
            "System mit Eingang",
            explanation=f"Gegeben: ẋ = f(x,u) mit f = [{', '.join(str(f) for f in f_funcs)}]"
        )
        
        # A-Matrix (∂f/∂x)
        A_symbolic = sp.Matrix([[sp.diff(f, x) for x in x_vars] for f in f_funcs])
        A_linearized = A_symbolic.subs(equilibrium_point)
        
        self.logger.add_step(
            "A-Matrix (∂f/∂x)",
            A_linearized,
            "A = ∂f/∂x am Gleichgewichtspunkt"
        )
        
        # B-Matrix (∂f/∂u)
        B_symbolic = sp.Matrix([[sp.diff(f, u) for u in u_vars] for f in f_funcs])
        B_linearized = B_symbolic.subs(equilibrium_point)
        
        self.logger.add_step(
            "B-Matrix (∂f/∂u)",
            B_linearized,
            "B = ∂f/∂u am Gleichgewichtspunkt"
        )
        
        # Ergebnisse anzeigen
        display_step_by_step(self.logger.get_steps(), "Linearisierung des Systems mit Eingang")
        
        # Zusammenfassung
        st.subheader("Linearisiertes System")
        st.latex(r"\\dot{\\mathbf{\\xi}} = \\mathbf{A} \\mathbf{\\xi} + \\mathbf{B} \\mathbf{\\upsilon}")
        st.markdown("mit $\\mathbf{\\xi} = \\mathbf{x} - \\mathbf{x}^*$ und $\\mathbf{\\upsilon} = \\mathbf{u} - \\mathbf{u}^*$")
        
        col1, col2 = st.columns(2)
        with col1:
            display_matrix(A_linearized, "Systemmatrix A")
        with col2:
            display_matrix(B_linearized, "Eingangsmatrix B")
    
    def _analyze_stability(self, eigenvals):
        """Analysiert die Stabilität basierend auf Eigenwerten"""
        real_parts = []
        
        for eigenval, multiplicity in eigenvals.items():
            if eigenval.is_real:
                real_parts.append(float(eigenval))
            else:
                # Komplexe Eigenwerte
                real_part = sp.re(eigenval)
                real_parts.append(float(real_part))
        
        max_real_part = max(real_parts)
        min_real_part = min(real_parts)
        
        if max_real_part < 0:
            return "asymptotisch stabil"
        elif min_real_part > 0:
            return "instabil"
        elif max_real_part == 0:
            return "marginal stabil (weitere Analyse nötig)"
        else:
            return "instabil"
