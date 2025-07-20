"""
Zustandstransformation
"""

import streamlit as st
import sympy as sp
from modules.base_module import BaseModule
from utils.display_utils import display_step_by_step, display_latex, display_matrix

class StateTransformationModule(BaseModule):
    """Modul für Zustandstransformationen"""
    
    def __init__(self):
        super().__init__(
            "Zustandstransformation",
            "Transformation der Zustandsraumdarstellung in Jordanische Normalform und Regelungsnormalform"
        )
    
    def render(self):
        self.display_description()
        
        tab1, tab2 = st.tabs([
            "Jordanische Normalform",
            "Regelungsnormalform"
        ])
        
        with tab1:
            self.jordan_normal_form()
        
        with tab2:
            self.controllable_canonical_form()
    
    def jordan_normal_form(self):
        """Transformation in Jordanische Normalform"""
        st.subheader("Transformation in Jordanische Normalform")
        
        # Eingabe
        st.markdown("**Eingabe der Systemmatrix A:**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            matrix_input = st.text_area(
                "Matrix A (eine Zeile pro Matrixzeile, Elemente durch Komma getrennt):",
                "0, 1\n-2, -3",
                height=100
            )
        
        with col2:
            st.markdown("**Beispiele:**")
            if st.button("Beispiel 1: 2×2 System"):
                st.session_state.matrix_example = "0, 1\n-2, -3"
            if st.button("Beispiel 2: 3×3 System"):
                st.session_state.matrix_example = "0, 1, 0\n0, 0, 1\n-6, -11, -6"
            if st.button("Beispiel 3: Mehrfache Eigenwerte"):
                st.session_state.matrix_example = "-1, 1\n0, -1"
        
        if st.button("Jordan-Normalform berechnen", key="jordan_calc"):
            try:
                # Matrix parsen
                A = self._parse_matrix(matrix_input)
                self._calculate_jordan_form(A)
            except Exception as e:
                st.error(f"Fehler bei der Eingabe: {str(e)}")
                st.info("Bitte überprüfen Sie das Format der Matrix.")
    
    def controllable_canonical_form(self):
        """Transformation in Regelungsnormalform"""
        st.subheader("Transformation in Regelungsnormalform (Steuerungsnormalform)")
        
        # Eingabe
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Systemmatrix A:**")
            matrix_A = st.text_area(
                "Matrix A:",
                "0, 1\n-2, -3",
                height=100,
                key="ccf_A"
            )
        
        with col2:
            st.markdown("**Eingangsmatrix B:**")
            matrix_B = st.text_area(
                "Matrix B:",
                "0\n1",
                height=100,
                key="ccf_B"
            )
        
        if st.button("Regelungsnormalform berechnen", key="ccf_calc"):
            try:
                A = self._parse_matrix(matrix_A)
                B = self._parse_matrix(matrix_B)
                self._calculate_controllable_canonical_form(A, B)
            except Exception as e:
                st.error(f"Fehler bei der Eingabe: {str(e)}")
    
    def _parse_matrix(self, matrix_str):
        """Parst einen Matrix-String zu SymPy Matrix"""
        lines = matrix_str.strip().split('\n')
        matrix_data = []
        
        for line in lines:
            row = [sp.sympify(x.strip()) for x in line.split(',')]
            matrix_data.append(row)
        
        return sp.Matrix(matrix_data)
    
    def _calculate_jordan_form(self, A):
        """Berechnet die Jordan-Normalform"""
        self.logger.clear()
        
        # Schritt 1: Eigenwerte berechnen
        self.logger.add_step(
            "Berechnung der Eigenwerte",
            explanation="Löse die charakteristische Gleichung det(A - λI) = 0"
        )
        
        lambda_sym = sp.Symbol('lambda')
        I = sp.eye(A.rows)
        char_poly = (A - lambda_sym * I).det()
        eigenvalues = sp.solve(char_poly, lambda_sym)
        
        self.logger.add_step(
            "Charakteristisches Polynom",
            char_poly,
            f"det(A - λI) = {char_poly}"
        )
        
        st.markdown("**Eigenwerte:**")
        for i, ev in enumerate(eigenvalues):
            st.latex(f"\\lambda_{{{i+1}}} = {sp.latex(ev)}")
        
        # Schritt 2: Eigenvektoren berechnen
        self.logger.add_step(
            "Berechnung der Eigenvektoren",
            explanation="Für jeden Eigenwert λᵢ löse (A - λᵢI)v = 0"
        )
        
        eigenvectors = []
        jordan_blocks = []
        
        for ev in eigenvalues:
            # Eigenvektor berechnen
            eigenspace_matrix = A - ev * I
            eigenspace_null = eigenspace_matrix.nullspace()
            
            if eigenspace_null:
                eigenvectors.extend(eigenspace_null)
                st.markdown(f"**Eigenvektor für λ = {ev}:**")
                for vec in eigenspace_null:
                    display_matrix(vec, f"v für λ = {ev}")
        
        # Schritt 3: Jordan-Matrix konstruieren
        try:
            P, J = A.jordan_form()
            
            self.logger.add_step(
                "Transformationsmatrix P",
                P,
                "Matrix P enthält die Jordan-Vektoren (Eigenvektoren und ggf. Hauptvektoren)"
            )
            
            self.logger.add_step(
                "Jordan-Matrix J",
                J,
                "Die Jordan-Normalform J = P⁻¹AP"
            )
            
            # Verifikation
            P_inv = P.inv()
            verification = P_inv * A * P
            
            self.logger.add_step(
                "Verifikation",
                verification,
                "Überprüfung: P⁻¹AP = J"
            )
            
            # Ergebnisse anzeigen
            display_step_by_step(self.logger.get_steps(), "Jordan-Normalform Transformation")
            
            # Zusammenfassung
            st.subheader("Ergebnis der Jordan-Transformation")
            
            col1, col2 = st.columns(2)
            with col1:
                display_matrix(P, "Transformationsmatrix P")
                display_matrix(P_inv, "Inverse Matrix P⁻¹")
            
            with col2:
                display_matrix(J, "Jordan-Matrix J")
                st.markdown("**Transformationsbeziehung:**")
                st.latex(r"J = P^{-1}AP")
                
        except Exception as e:
            st.error(f"Fehler bei der Jordan-Zerlegung: {str(e)}")
            st.info("Möglicherweise ist die Matrix nicht über den rationalen Zahlen diagonalisierbar.")
    
    def _calculate_controllable_canonical_form(self, A, B):
        """Berechnet die Regelungsnormalform"""
        self.logger.clear()
        
        # Schritt 1: Steuerbarkeitsmatrix berechnen
        self.logger.add_step(
            "Steuerbarkeitsmatrix berechnen",
            explanation="Berechne Q = [B, AB, A²B, ..., Aⁿ⁻¹B]"
        )
        
        n = A.rows
        Q_blocks = [B]
        
        current_block = B
        for i in range(1, n):
            current_block = A * current_block
            Q_blocks.append(current_block)
        
        Q = sp.Matrix.hstack(*Q_blocks)
        
        self.logger.add_step(
            "Steuerbarkeitsmatrix Q",
            Q,
            f"Q = [B, AB, A²B, ..., A^{n-1}B]"
        )
        
        # Schritt 2: Steuerbarkeit prüfen
        Q_det = Q.det()
        
        if Q_det == 0:
            st.error("Das System ist nicht steuerbar! Regelungsnormalform existiert nicht.")
            return
        
        self.logger.add_step(
            "Steuerbarkeitstest",
            Q_det,
            f"det(Q) = {Q_det} ≠ 0 → System ist steuerbar"
        )
        
        # Schritt 3: Charakteristisches Polynom
        lambda_sym = sp.Symbol('lambda')
        I = sp.eye(n)
        char_poly = (A - lambda_sym * I).det()
        char_poly_expanded = sp.expand(char_poly)
        
        # Koeffizienten extrahieren
        coeffs = sp.Poly(char_poly_expanded, lambda_sym).all_coeffs()
        coeffs.reverse()  # [a₀, a₁, ..., aₙ₋₁, 1]
        
        self.logger.add_step(
            "Charakteristisches Polynom",
            char_poly_expanded,
            f"λⁿ + aₙ₋₁λⁿ⁻¹ + ... + a₁λ + a₀ = 0"
        )
        
        # Schritt 4: Regelungsnormalform konstruieren
        A_c = sp.zeros(n, n)
        
        # Letzte Zeile: negative Koeffizienten des char. Polynoms
        for j in range(n):
            if j < len(coeffs) - 1:  # -1 weil wir den führenden Koeffizienten (1) ignorieren
                A_c[n-1, j] = -coeffs[j]
        
        # Obere Nebendiagonale: Einsen
        for i in range(n-1):
            A_c[i, i+1] = 1
        
        B_c = sp.zeros(n, 1)
        B_c[n-1, 0] = 1
        
        self.logger.add_step(
            "Regelungsnormalform A_c",
            A_c,
            "Die Systemmatrix in Regelungsnormalform"
        )
        
        self.logger.add_step(
            "Regelungsnormalform B_c",
            B_c,
            "Die Eingangsmatrix in Regelungsnormalform"
        )
        
        # Schritt 5: Transformationsmatrix berechnen
        # T = Q * Q_c^(-1), wobei Q_c die Steuerbarkeitsmatrix der Regelungsnormalform ist
        Q_c_blocks = [B_c]
        current_block = B_c
        for i in range(1, n):
            current_block = A_c * current_block
            Q_c_blocks.append(current_block)
        
        Q_c = sp.Matrix.hstack(*Q_c_blocks)
        T = Q * Q_c.inv()
        
        self.logger.add_step(
            "Transformationsmatrix T",
            T,
            "T = Q * Q_c⁻¹ transformiert zur Regelungsnormalform"
        )
        
        # Verifikation
        T_inv = T.inv()
        A_transformed = T_inv * A * T
        B_transformed = T_inv * B
        
        self.logger.add_step(
            "Verifikation A_c",
            A_transformed,
            "Überprüfung: T⁻¹AT = A_c"
        )
        
        self.logger.add_step(
            "Verifikation B_c",
            B_transformed,
            "Überprüfung: T⁻¹B = B_c"
        )
        
        # Ergebnisse anzeigen
        display_step_by_step(self.logger.get_steps(), "Transformation zur Regelungsnormalform")
        
        # Zusammenfassung
        st.subheader("Ergebnis der Regelungsnormalform-Transformation")
        
        col1, col2 = st.columns(2)
        with col1:
            display_matrix(A_c, "Systemmatrix A_c (Regelungsnormalform)")
            display_matrix(B_c, "Eingangsmatrix B_c (Regelungsnormalform)")
        
        with col2:
            display_matrix(T, "Transformationsmatrix T")
            st.markdown("**Transformationsbeziehungen:**")
            st.latex(r"A_c = T^{-1}AT")
            st.latex(r"B_c = T^{-1}B")
