"""
LTI-Systeme und Zustandsraumdarstellung
"""

import streamlit as st
import sympy as sp
import numpy as np
from modules.base_module import BaseModule
from utils.display_utils import display_step_by_step, display_latex, display_matrix, plot_system_response
import control

class LTISystemsModule(BaseModule):
    """Modul für LTI-Systeme und Zustandsraumdarstellung"""
    
    def __init__(self):
        super().__init__(
            "LTI-Systeme & Zustandsraumdarstellung",
            "Bestimmung der Zustandsraumdarstellung aus Differentialgleichungen und Analyse von LTI-Systemen"
        )
    
    def render(self):
        self.display_description()
        
        tab1, tab2, tab3 = st.tabs([
            "Differentialgleichung → Zustandsraum",
            "Übertragungsfunktion → Zustandsraum", 
            "Systemanalyse"
        ])
        
        with tab1:
            self.differential_to_state_space()
        
        with tab2:
            self.transfer_to_state_space()
        
        with tab3:
            self.system_analysis()
    
    def differential_to_state_space(self):
        """Umwandlung von Differentialgleichung zu Zustandsraumdarstellung"""
        st.subheader("Differentialgleichung → Zustandsraumdarstellung")
        
        # Eingabe
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Systemparameter:**")
            order = st.selectbox("Systemordnung:", [2, 3, 4], index=0)
            
        with col2:
            st.markdown("**Eingabeformat:**")
            input_format = st.selectbox("Format:", ["Symbolisch", "Numerisch"], index=0)
        
        if input_format == "Symbolisch":
            self._symbolic_differential_input(order)
        else:
            self._numeric_differential_input(order)
    
    def _symbolic_differential_input(self, order: int):
        """Symbolische Eingabe der Differentialgleichung"""
        st.markdown("**Differentialgleichung eingeben:**")
        
        if order == 2:
            st.latex(r"a_2 \ddot{y}(t) + a_1 \dot{y}(t) + a_0 y(t) = b_1 \dot{u}(t) + b_0 u(t)")
            
            col1, col2 = st.columns(2)
            with col1:
                a2 = st.text_input("Koeffizient a₂:", "1")
                a1 = st.text_input("Koeffizient a₁:", "3")
                a0 = st.text_input("Koeffizient a₀:", "2")
            
            with col2:
                b1 = st.text_input("Koeffizient b₁:", "0")
                b0 = st.text_input("Koeffizient b₀:", "1")
        
        elif order == 3:
            st.latex(r"a_3 \dddot{y}(t) + a_2 \ddot{y}(t) + a_1 \dot{y}(t) + a_0 y(t) = b_2 \ddot{u}(t) + b_1 \dot{u}(t) + b_0 u(t)")
            
            col1, col2 = st.columns(2)
            with col1:
                a3 = st.text_input("Koeffizient a₃:", "1")
                a2 = st.text_input("Koeffizient a₂:", "4")
                a1 = st.text_input("Koeffizient a₁:", "5")
                a0 = st.text_input("Koeffizient a₀:", "2")
            
            with col2:
                b2 = st.text_input("Koeffizient b₂:", "0")
                b1 = st.text_input("Koeffizient b₁:", "1")
                b0 = st.text_input("Koeffizient b₀:", "1")
        
        if st.button("Zustandsraumdarstellung berechnen", key="calc_ss"):
            try:
                if order == 2:
                    coeffs_a = [sp.sympify(a2), sp.sympify(a1), sp.sympify(a0)]
                    coeffs_b = [sp.sympify(b1), sp.sympify(b0)]
                elif order == 3:
                    coeffs_a = [sp.sympify(a3), sp.sympify(a2), sp.sympify(a1), sp.sympify(a0)]
                    coeffs_b = [sp.sympify(b2), sp.sympify(b1), sp.sympify(b0)]
                
                self._calculate_state_space_from_diff(coeffs_a, coeffs_b, order)
                
            except Exception as e:
                st.error(f"Fehler bei der Berechnung: {str(e)}")
    
    def _calculate_state_space_from_diff(self, coeffs_a, coeffs_b, order):
        """Berechnet die Zustandsraumdarstellung aus Differentialgleichungskoeffizienten"""
        self.logger.clear()
        
        # Schritt 1: Normierung
        self.logger.add_step(
            "Normierung der Differentialgleichung",
            explanation="Teile durch den Koeffizienten der höchsten Ableitung, um eine normierte Form zu erhalten."
        )
        
        a_lead = coeffs_a[0]
        coeffs_a_norm = [coeff / a_lead for coeff in coeffs_a]
        coeffs_b_norm = [coeff / a_lead for coeff in coeffs_b]
        
        # Schritt 2: Zustandsvariablen definieren
        if order == 2:
            x1, x2 = sp.symbols('x1 x2')
            self.logger.add_step(
                "Definition der Zustandsvariablen",
                explanation="Wähle die Zustandsvariablen als: x₁ = y, x₂ = ẏ"
            )
            
            # Systemmatrizen
            A = sp.Matrix([
                [0, 1],
                [-coeffs_a_norm[2], -coeffs_a_norm[1]]
            ])
            
            # B-Matrix abhängig von b-Koeffizienten
            if len(coeffs_b_norm) > 1 and coeffs_b_norm[0] != 0:
                B = sp.Matrix([
                    [coeffs_b_norm[0]],
                    [coeffs_b_norm[1] - coeffs_a_norm[1] * coeffs_b_norm[0]]
                ])
            else:
                B = sp.Matrix([
                    [0],
                    [coeffs_b_norm[-1]]
                ])
            
            C = sp.Matrix([[1, 0]])
            D = sp.Matrix([[coeffs_b_norm[0] if len(coeffs_b_norm) > 1 else 0]])
        
        elif order == 3:
            x1, x2, x3 = sp.symbols('x1 x2 x3')
            self.logger.add_step(
                "Definition der Zustandsvariablen",
                explanation="Wähle die Zustandsvariablen als: x₁ = y, x₂ = ẏ, x₃ = ÿ"
            )
            
            A = sp.Matrix([
                [0, 1, 0],
                [0, 0, 1],
                [-coeffs_a_norm[3], -coeffs_a_norm[2], -coeffs_a_norm[1]]
            ])
            
            if len(coeffs_b_norm) > 2 and coeffs_b_norm[0] != 0:
                B = sp.Matrix([
                    [coeffs_b_norm[0]],
                    [coeffs_b_norm[1] - coeffs_a_norm[1] * coeffs_b_norm[0]],
                    [coeffs_b_norm[2] - coeffs_a_norm[2] * coeffs_b_norm[0] - coeffs_a_norm[1] * (coeffs_b_norm[1] - coeffs_a_norm[1] * coeffs_b_norm[0])]
                ])
            else:
                B = sp.Matrix([
                    [0],
                    [0],
                    [coeffs_b_norm[-1]]
                ])
            
            C = sp.Matrix([[1, 0, 0]])
            D = sp.Matrix([[coeffs_b_norm[0] if len(coeffs_b_norm) > 2 else 0]])
        
        # Schritt 3: Systemmatrizen anzeigen
        self.logger.add_step(
            "Systemmatrix A",
            A,
            "Die Systemmatrix A beschreibt die Dynamik des Systems."
        )
        
        self.logger.add_step(
            "Eingangsmatrix B",
            B,
            "Die Eingangsmatrix B beschreibt, wie der Eingang auf die Zustände wirkt."
        )
        
        self.logger.add_step(
            "Ausgangsmatrix C",
            C,
            "Die Ausgangsmatrix C beschreibt, welche Zustände gemessen werden."
        )
        
        self.logger.add_step(
            "Durchgangsmatrix D",
            D,
            "Die Durchgangsmatrix D beschreibt den direkten Durchgriff vom Eingang zum Ausgang."
        )
        
        # Ergebnisse anzeigen
        display_step_by_step(self.logger.get_steps(), "Herleitung der Zustandsraumdarstellung")
        
        # Zustandsraumdarstellung zusammenfassen
        st.subheader("Zustandsraumdarstellung")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Zustandsgleichung:**")
            st.latex(r"\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} u(t)")
            display_matrix(A, "Matrix A")
            display_matrix(B, "Matrix B")
        
        with col2:
            st.markdown("**Ausgangsgleichung:**")
            st.latex(r"y(t) = \mathbf{C} \mathbf{x}(t) + \mathbf{D} u(t)")
            display_matrix(C, "Matrix C")
            display_matrix(D, "Matrix D")
    
    def transfer_to_state_space(self):
        """Umwandlung von Übertragungsfunktion zu Zustandsraumdarstellung"""
        st.subheader("Übertragungsfunktion → Zustandsraumdarstellung")
        
        # Eingabe
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Zählerpolynom:**")
            numerator = st.text_input(
                "Zähler (absteigende Potenzen):",
                "1",
                help="z.B. '2*s + 1' oder '1'"
            )
        
        with col2:
            st.markdown("**Nennerpolynom:**")
            denominator = st.text_input(
                "Nenner (absteigende Potenzen):",
                "s**2 + 3*s + 2",
                help="z.B. 's**2 + 3*s + 2'"
            )
        
        realization = st.selectbox(
            "Realisierungsform:",
            ["Steuerungsnormalform", "Beobachtungsnormalform", "Modale Form"],
            index=0
        )
        
        if st.button("Zustandsraumdarstellung berechnen", key="tf_to_ss"):
            try:
                self._tf_to_state_space(numerator, denominator, realization)
            except Exception as e:
                st.error(f"Fehler bei der Umwandlung: {str(e)}")
    
    def system_analysis(self):
        """Systemanalyse"""
        st.subheader("Systemanalyse")
        
        # Eingabe der Systemmatrizen
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Systemmatrix A:**")
            matrix_A = st.text_area(
                "Matrix A:",
                "0, 1\n-2, -3",
                height=100,
                key="analysis_A"
            )
            
            st.markdown("**Ausgangsmatrix C:**")
            matrix_C = st.text_area(
                "Matrix C:",
                "1, 0",
                height=50,
                key="analysis_C"
            )
        
        with col2:
            st.markdown("**Eingangsmatrix B:**")
            matrix_B = st.text_area(
                "Matrix B:",
                "0\n1",
                height=100,
                key="analysis_B"
            )
            
            st.markdown("**Durchgangsmatrix D:**")
            matrix_D = st.text_area(
                "Matrix D:",
                "0",
                height=50,
                key="analysis_D"
            )
        
        analyses = st.multiselect(
            "Gewünschte Analysen:",
            ["Stabilität", "Steuerbarkeit", "Beobachtbarkeit", "Eigenwerte", "Sprungantwort"],
            default=["Stabilität", "Steuerbarkeit", "Beobachtbarkeit"]
        )
        
        if st.button("Systemanalyse durchführen", key="sys_analysis"):
            try:
                A = self._parse_matrix(matrix_A)
                B = self._parse_matrix(matrix_B)
                C = self._parse_matrix(matrix_C)
                D = self._parse_matrix(matrix_D)
                self._perform_system_analysis(A, B, C, D, analyses)
            except Exception as e:
                st.error(f"Fehler bei der Analyse: {str(e)}")
    
    def _tf_to_state_space(self, num_str, den_str, realization):
        """Wandelt Übertragungsfunktion in Zustandsraumdarstellung um"""
        self.logger.clear()
        
        s = sp.Symbol('s')
        numerator = sp.sympify(num_str)
        denominator = sp.sympify(den_str)
        
        # Übertragungsfunktion
        G_s = numerator / denominator
        
        self.logger.add_step(
            "Übertragungsfunktion",
            G_s,
            f"G(s) = ({num_str}) / ({den_str})"
        )
        
        # Polynome in Standardform bringen
        num_poly = sp.Poly(numerator, s)
        den_poly = sp.Poly(denominator, s)
        
        num_coeffs = num_poly.all_coeffs()
        den_coeffs = den_poly.all_coeffs()
        
        # Auf gleiche Länge bringen (mit führenden Nullen)
        order = len(den_coeffs) - 1
        
        if len(num_coeffs) < len(den_coeffs):
            num_coeffs = [0] * (len(den_coeffs) - len(num_coeffs)) + list(num_coeffs)
        
        self.logger.add_step(
            "Polynomkoeffizienten",
            explanation=f"Zähler: {num_coeffs}, Nenner: {den_coeffs}"
        )
        
        if realization == "Steuerungsnormalform":
            A, B, C, D = self._controllable_canonical_realization(den_coeffs, num_coeffs)
        elif realization == "Beobachtungsnormalform":
            A, B, C, D = self._observable_canonical_realization(den_coeffs, num_coeffs)
        else:  # Modale Form
            st.info("Modale Form wird in einer zukünftigen Version implementiert.")
            return
        
        self.logger.add_step(
            f"Systemmatrix A ({realization})",
            A,
            f"A-Matrix in {realization}"
        )
        
        self.logger.add_step(
            f"Eingangsmatrix B ({realization})",
            B,
            f"B-Matrix in {realization}"
        )
        
        self.logger.add_step(
            f"Ausgangsmatrix C ({realization})",
            C,
            f"C-Matrix in {realization}"
        )
        
        self.logger.add_step(
            f"Durchgangsmatrix D ({realization})",
            D,
            f"D-Matrix in {realization}"
        )
        
        # Ergebnisse anzeigen
        display_step_by_step(self.logger.get_steps(), f"Übertragungsfunktion → {realization}")
        
        # Zusammenfassung
        st.subheader(f"Zustandsraumdarstellung ({realization})")
        
        col1, col2 = st.columns(2)
        with col1:
            display_matrix(A, "Matrix A")
            display_matrix(B, "Matrix B")
        with col2:
            display_matrix(C, "Matrix C")
            display_matrix(D, "Matrix D")
    
    def _controllable_canonical_realization(self, den_coeffs, num_coeffs):
        """Erstellt Steuerungsnormalform"""
        n = len(den_coeffs) - 1
        
        # Normiere auf führenden Koeffizienten
        a0 = den_coeffs[0]
        den_norm = [coeff / a0 for coeff in den_coeffs]
        num_norm = [coeff / a0 for coeff in num_coeffs]
        
        # A-Matrix
        A = sp.zeros(n, n)
        for i in range(n-1):
            A[i, i+1] = 1
        for j in range(n):
            A[n-1, j] = -den_norm[n-j]
        
        # B-Matrix
        B = sp.zeros(n, 1)
        B[n-1, 0] = 1
        
        # C-Matrix
        C = sp.zeros(1, n)
        for j in range(n):
            if j < len(num_norm):
                C[0, j] = num_norm[n-j] - num_norm[0] * den_norm[n-j]
        
        # D-Matrix
        D = sp.Matrix([[num_norm[0]]])
        
        return A, B, C, D
    
    def _observable_canonical_realization(self, den_coeffs, num_coeffs):
        """Erstellt Beobachtungsnormalform"""
        n = len(den_coeffs) - 1
        
        # Normiere auf führenden Koeffizienten
        a0 = den_coeffs[0]
        den_norm = [coeff / a0 for coeff in den_coeffs]
        num_norm = [coeff / a0 for coeff in num_coeffs]
        
        # A-Matrix (Transponierte der Steuerungsnormalform)
        A = sp.zeros(n, n)
        for i in range(1, n):
            A[i, i-1] = 1
        for i in range(n):
            A[i, n-1] = -den_norm[n-i]
        
        # B-Matrix
        B = sp.zeros(n, 1)
        for i in range(n):
            if i < len(num_norm):
                B[i, 0] = num_norm[n-i] - num_norm[0] * den_norm[n-i]
        
        # C-Matrix
        C = sp.zeros(1, n)
        C[0, n-1] = 1
        
        # D-Matrix
        D = sp.Matrix([[num_norm[0]]])
        
        return A, B, C, D
    
    def _perform_system_analysis(self, A, B, C, D, analyses):
        """Führt verschiedene Systemanalysen durch"""
        self.logger.clear()
        
        n = A.rows
        
        if "Eigenwerte" in analyses:
            eigenvals = A.eigenvals()
            self.logger.add_step(
                "Eigenwerte",
                explanation=f"Eigenwerte der Matrix A: {list(eigenvals.keys())}"
            )
        
        if "Stabilität" in analyses:
            stability = self._analyze_stability_ss(A)
            self.logger.add_step(
                "Stabilitätsanalyse",
                explanation=f"Stabilität: {stability}"
            )
        
        if "Steuerbarkeit" in analyses:
            controllability = self._check_controllability(A, B)
            self.logger.add_step(
                "Steuerbarkeitsanalyse",
                explanation=f"Steuerbarkeit: {controllability}"
            )
        
        if "Beobachtbarkeit" in analyses:
            observability = self._check_observability(A, C)
            self.logger.add_step(
                "Beobachtbarkeitsanalyse",
                explanation=f"Beobachtbarkeit: {observability}"
            )
        
        # Ergebnisse anzeigen
        display_step_by_step(self.logger.get_steps(), "Systemanalyse")
        
        # Detaillierte Ergebnisse
        if "Sprungantwort" in analyses:
            st.subheader("Sprungantwort")
            self._plot_step_response_ss(A, B, C, D)
    
    def _analyze_stability_ss(self, A):
        """Stabilitätsanalyse für Zustandsraumdarstellung"""
        eigenvals = A.eigenvals()
        real_parts = []
        
        for eigenval in eigenvals.keys():
            if eigenval.is_real:
                real_parts.append(float(eigenval))
            else:
                real_part = sp.re(eigenval)
                real_parts.append(float(real_part.evalf()))
        
        max_real_part = max(real_parts) if real_parts else 0
        
        if max_real_part < -1e-10:
            return "asymptotisch stabil"
        elif abs(max_real_part) < 1e-10:
            return "marginal stabil"
        else:
            return "instabil"
    
    def _check_controllability(self, A, B):
        """Überprüft Steuerbarkeit"""
        n = A.rows
        Q_blocks = [B]
        
        current_block = B
        for i in range(1, n):
            current_block = A * current_block
            Q_blocks.append(current_block)
        
        Q = sp.Matrix.hstack(*Q_blocks)
        rank_Q = Q.rank()
        
        if rank_Q == n:
            return f"vollständig steuerbar (rang(Q) = {rank_Q} = {n})"
        else:
            return f"nicht vollständig steuerbar (rang(Q) = {rank_Q} < {n})"
    
    def _check_observability(self, A, C):
        """Überprüft Beobachtbarkeit"""
        n = A.rows
        O_blocks = [C]
        
        current_block = C
        for i in range(1, n):
            current_block = current_block * A
            O_blocks.append(current_block)
        
        O = sp.Matrix.vstack(*O_blocks)
        rank_O = O.rank()
        
        if rank_O == n:
            return f"vollständig beobachtbar (rang(O) = {rank_O} = {n})"
        else:
            return f"nicht vollständig beobachtbar (rang(O) = {rank_O} < {n})"
    
    def _plot_step_response_ss(self, A, B, C, D):
        """Plottet Sprungantwort für Zustandsraumdarstellung"""
        try:
            # Vereinfachte numerische Simulation
            import numpy as np
            from scipy.integrate import odeint
            
            # Konvertiere zu numerischen Matrizen
            A_num = np.array(A.evalf()).astype(float)
            B_num = np.array(B.evalf()).astype(float)
            C_num = np.array(C.evalf()).astype(float)
            D_num = np.array(D.evalf()).astype(float)
            
            def system_ode(x, t):
                u = 1  # Sprungeingang
                return A_num @ x + B_num.flatten() * u
            
            t = np.linspace(0, 10, 1000)
            x0 = np.zeros(A.rows)
            
            # Löse ODE
            x_trajectory = odeint(system_ode, x0, t)
            
            # Berechne Ausgang
            y = []
            for i, x_val in enumerate(x_trajectory):
                u_val = 1  # Sprungeingang
                y_val = float(C_num @ x_val + D_num * u_val)
                y.append(y_val)
            
            plot_system_response(t, y, "Sprungantwort", "Zeit t [s]", "Ausgangsgröße y(t)")
            
        except Exception as e:
            st.error(f"Sprungantwort konnte nicht berechnet werden: {str(e)}")
            st.info("Stellen Sie sicher, dass alle Matrixelemente numerisch auswertbar sind.")
    
    def _parse_matrix(self, matrix_str):
        """Parst einen Matrix-String zu SymPy Matrix"""
        lines = matrix_str.strip().split('\n')
        matrix_data = []
        
        for line in lines:
            row = [sp.sympify(x.strip()) for x in line.split(',')]
            matrix_data.append(row)
        
        return sp.Matrix(matrix_data)
