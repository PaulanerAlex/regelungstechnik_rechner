"""
Blockschaltbild-Umformung
"""

import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.base_module import BaseModule
from utils.display_utils import display_step_by_step, display_latex
from utils.safe_sympify import safe_sympify

class BlockDiagramModule(BaseModule):
    """Modul für Blockschaltbild-Umformungen und -Vereinfachungen"""
    
    def __init__(self):
        super().__init__(
            "Blockschaltbild-Umformung",
            "Umformen und Vereinfachen von Blockschaltbildern für Regelungssysteme"
        )
    
    def render(self):
        self.display_description()
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Grundverknüpfungen",
            "Umformungsregeln",
            "Interaktiver Rechner",
            "Komplexe Strukturen"
        ])
        
        with tab1:
            self.basic_connections()
        
        with tab2:
            self.transformation_rules()
        
        with tab3:
            self.interactive_calculator()
        
        with tab4:
            self.complex_structures()
    
    def basic_connections(self):
        """Grundlegende Blockverknüpfungen"""
        st.subheader("Grundlegende Blockverknüpfungen")
        
        connection_type = st.selectbox(
            "Wählen Sie eine Verknüpfung:",
            [
                "Reihenschaltung",
                "Parallelschaltung", 
                "Rückkopplung (negativ)",
                "Rückkopplung (positiv)",
                "Vorwärtskopplung"
            ]
        )
        
        if connection_type == "Reihenschaltung":
            self._show_series_connection()
        elif connection_type == "Parallelschaltung":
            self._show_parallel_connection()
        elif connection_type == "Rückkopplung (negativ)":
            self._show_negative_feedback()
        elif connection_type == "Rückkopplung (positiv)":
            self._show_positive_feedback()
        elif connection_type == "Vorwärtskopplung":
            self._show_feedforward()
    
    def _show_series_connection(self):
        """Reihenschaltung"""
        st.markdown("### Reihenschaltung")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Blockschaltbild:**")
            st.text("""
            U(s) → [G₁(s)] → [G₂(s)] → Y(s)
            """)
            
            st.markdown("**Gesamtübertragungsfunktion:**")
            st.latex(r"G_{ges}(s) = G_1(s) \cdot G_2(s)")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Ausgangssignal des ersten Blocks = Eingangssignal des zweiten Blocks
            - Übertragungsfunktionen werden multipliziert
            - Reihenfolge ist vertauschbar (kommutativ)
            """)
        
        with col2:
            # Eingabe der Übertragungsfunktionen
            st.markdown("**Beispielrechnung:**")
            
            g1_input = st.text_input("G₁(s) =", value="2/(s+1)", key="series_g1")
            g2_input = st.text_input("G₂(s) =", value="3/(s+2)", key="series_g2")
            
            if st.button("Berechnen", key="series_calc"):
                try:
                    s = sp.Symbol('s')
                    G1 = safe_sympify(g1_input)
                    G2 = safe_sympify(g2_input)
                    
                    G_total = G1 * G2
                    G_total_simplified = sp.simplify(G_total)
                    
                    st.latex(f"G_1(s) = {sp.latex(G1)}")
                    st.latex(f"G_2(s) = {sp.latex(G2)}")
                    st.latex(f"G_{{ges}}(s) = {sp.latex(G_total_simplified)}")
                    
                    # Pol-Nullstellen
                    poles_g1 = sp.solve(sp.denom(G1), s)
                    poles_g2 = sp.solve(sp.denom(G2), s)
                    zeros_g1 = sp.solve(sp.numer(G1), s)
                    zeros_g2 = sp.solve(sp.numer(G2), s)
                    
                    st.markdown("**Pole und Nullstellen:**")
                    st.write(f"Pole G₁: {poles_g1}")
                    st.write(f"Pole G₂: {poles_g2}")
                    st.write(f"Nullstellen G₁: {zeros_g1}")
                    st.write(f"Nullstellen G₂: {zeros_g2}")
                    
                except Exception as e:
                    st.error(f"Fehler bei der Berechnung: {e}")
    
    def _show_parallel_connection(self):
        """Parallelschaltung"""
        st.markdown("### Parallelschaltung")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Blockschaltbild:**")
            st.text("""
                    ┌→ [G₁(s)] →┐
            U(s) →→ ┤            ├ + → Y(s)
                    └→ [G₂(s)] →┘
            """)
            
            st.markdown("**Gesamtübertragungsfunktion:**")
            st.latex(r"G_{ges}(s) = G_1(s) + G_2(s)")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Gleiches Eingangssignal für beide Blöcke
            - Ausgangssignale werden addiert
            - Übertragungsfunktionen werden addiert
            """)
        
        with col2:
            st.markdown("**Beispielrechnung:**")
            
            g1_input = st.text_input("G₁(s) =", value="1/(s+1)", key="parallel_g1")
            g2_input = st.text_input("G₂(s) =", value="2/(s+2)", key="parallel_g2")
            
            if st.button("Berechnen", key="parallel_calc"):
                try:
                    s = sp.Symbol('s')
                    G1 = safe_sympify(g1_input)
                    G2 = safe_sympify(g2_input)
                    
                    G_total = G1 + G2
                    G_total_simplified = sp.simplify(G_total)
                    
                    st.latex(f"G_1(s) = {sp.latex(G1)}")
                    st.latex(f"G_2(s) = {sp.latex(G2)}")
                    st.latex(f"G_{{ges}}(s) = {sp.latex(G_total_simplified)}")
                    
                except Exception as e:
                    st.error(f"Fehler bei der Berechnung: {e}")
    
    def _show_negative_feedback(self):
        """Negative Rückkopplung"""
        st.markdown("### Negative Rückkopplung (Standard-Regelkreis)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Blockschaltbild:**")
            st.text("""
            W(s) → + → [G₀(s)] → Y(s)
                   ↑              ↓
                   - ← [H(s)] ←←←←
            """)
            
            st.markdown("**Gesamtübertragungsfunktion:**")
            st.latex(r"G_{ges}(s) = \frac{G_0(s)}{1 + G_0(s) \cdot H(s)}")
            
            st.markdown("**Spezialfall H(s) = 1:**")
            st.latex(r"G_{ges}(s) = \frac{G_0(s)}{1 + G_0(s)}")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Stabilisierender Effekt
            - Reduziert Störeinflüsse
            - Verbessert Regelgenauigkeit
            - Kann instabil werden bei zu hoher Verstärkung
            """)
        
        with col2:
            st.markdown("**Beispielrechnung:**")
            
            g0_input = st.text_input("G₀(s) =", value="10/(s+1)", key="feedback_g0")
            h_input = st.text_input("H(s) =", value="1", key="feedback_h")
            
            if st.button("Berechnen", key="feedback_calc"):
                try:
                    s = sp.Symbol('s')
                    G0 = safe_sympify(g0_input)
                    H = safe_sympify(h_input)
                    
                    # Führungsübertragungsfunktion
                    G_w = G0 / (1 + G0 * H)
                    G_w_simplified = sp.simplify(G_w)
                    
                    # Charakteristisches Polynom
                    char_poly = 1 + G0 * H
                    char_poly_simplified = sp.simplify(char_poly)
                    
                    st.latex(f"G_0(s) = {sp.latex(G0)}")
                    st.latex(f"H(s) = {sp.latex(H)}")
                    st.latex(f"G_w(s) = {sp.latex(G_w_simplified)}")
                    
                    st.markdown("**Charakteristisches Polynom:**")
                    st.latex(f"1 + G_0(s)H(s) = {sp.latex(char_poly_simplified)}")
                    
                    # Stabilitätsanalyse
                    poles = sp.solve(char_poly_simplified, s)
                    st.markdown("**Pole des geschlossenen Kreises:**")
                    for i, pole in enumerate(poles):
                        real_part = sp.re(pole)
                        st.write(f"Pol {i+1}: {pole}")
                        if real_part < 0:
                            st.success(f"✓ Pol {i+1} liegt in der linken Halbebene (stabil)")
                        elif real_part > 0:
                            st.error(f"✗ Pol {i+1} liegt in der rechten Halbebene (instabil)")
                        else:
                            st.warning(f"⚠ Pol {i+1} liegt auf der imaginären Achse (Grenzstabilität)")
                    
                except Exception as e:
                    st.error(f"Fehler bei der Berechnung: {e}")
    
    def _show_positive_feedback(self):
        """Positive Rückkopplung"""
        st.markdown("### Positive Rückkopplung")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Blockschaltbild:**")
            st.text("""
            W(s) → + → [G₀(s)] → Y(s)
                   ↑              ↓
                   + ← [H(s)] ←←←←
            """)
            
            st.markdown("**Gesamtübertragungsfunktion:**")
            st.latex(r"G_{ges}(s) = \frac{G_0(s)}{1 - G_0(s) \cdot H(s)}")
            
            st.warning("⚠️ **Achtung**: Positive Rückkopplung kann zu Instabilität führen!")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Verstärkender Effekt
            - Kann zu Oszillationen führen
            - Instabil wenn G₀(s)·H(s) = 1
            - Selten in Regelungsanwendungen
            """)
        
        with col2:
            st.markdown("**Beispielrechnung:**")
            
            g0_input = st.text_input("G₀(s) =", value="2/(s+1)", key="pos_feedback_g0")
            h_input = st.text_input("H(s) =", value="0.5", key="pos_feedback_h")
            
            if st.button("Berechnen", key="pos_feedback_calc"):
                try:
                    s = sp.Symbol('s')
                    G0 = safe_sympify(g0_input)
                    H = safe_sympify(h_input)
                    
                    G_total = G0 / (1 - G0 * H)
                    G_total_simplified = sp.simplify(G_total)
                    
                    # Charakteristisches Polynom
                    char_poly = 1 - G0 * H
                    
                    st.latex(f"G_0(s) = {sp.latex(G0)}")
                    st.latex(f"H(s) = {sp.latex(H)}")
                    st.latex(f"G_{{ges}}(s) = {sp.latex(G_total_simplified)}")
                    
                    st.markdown("**Charakteristisches Polynom:**")
                    st.latex(f"1 - G_0(s)H(s) = {sp.latex(sp.simplify(char_poly))}")
                    
                    # Stabilitätsprüfung
                    poles = sp.solve(char_poly, s)
                    st.markdown("**Pole des geschlossenen Kreises:**")
                    stable = True
                    for i, pole in enumerate(poles):
                        real_part = sp.re(pole)
                        st.write(f"Pol {i+1}: {pole}")
                        if real_part >= 0:
                            stable = False
                    
                    if stable:
                        st.success("✓ System ist stabil")
                    else:
                        st.error("✗ System ist instabil!")
                    
                except Exception as e:
                    st.error(f"Fehler bei der Berechnung: {e}")
    
    def _show_feedforward(self):
        """Vorwärtskopplung"""
        st.markdown("### Vorwärtskopplung")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Blockschaltbild:**")
            st.text("""
            W(s) →→ [F(s)] →→ + → Y(s)
                               ↑
            W(s) → + → [G₀(s)] → 
                   ↑              
                   - ← [H(s)] ←←
            """)
            
            st.markdown("**Gesamtübertragungsfunktion:**")
            st.latex(r"G_{ges}(s) = \frac{G_0(s)}{1 + G_0(s)H(s)} + F(s)")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Kompensiert Störungen
            - Verbessert Führungsverhalten
            - Reduziert Regelabweichung
            - Benötigt Kenntnis der Störung
            """)
        
        with col2:
            st.markdown("**Beispielrechnung:**")
            
            g0_input = st.text_input("G₀(s) =", value="1/(s+1)", key="ff_g0")
            h_input = st.text_input("H(s) =", value="1", key="ff_h")
            f_input = st.text_input("F(s) =", value="s/(s+1)", key="ff_f")
            
            if st.button("Berechnen", key="ff_calc"):
                try:
                    s = sp.Symbol('s')
                    G0 = safe_sympify(g0_input)
                    H = safe_sympify(h_input)
                    F = safe_sympify(f_input)
                    
                    G_feedback = G0 / (1 + G0 * H)
                    G_total = G_feedback + F
                    G_total_simplified = sp.simplify(G_total)
                    
                    st.latex(f"G_0(s) = {sp.latex(G0)}")
                    st.latex(f"H(s) = {sp.latex(H)}")
                    st.latex(f"F(s) = {sp.latex(F)}")
                    st.latex(f"G_{{ges}}(s) = {sp.latex(G_total_simplified)}")
                    
                except Exception as e:
                    st.error(f"Fehler bei der Berechnung: {e}")
    
    def transformation_rules(self):
        """Umformungsregeln für Blockschaltbilder"""
        st.subheader("Umformungsregeln für Blockschaltbilder")
        
        rule_type = st.selectbox(
            "Wählen Sie eine Umformungsregel:",
            [
                "Verzweigungspunkt verschieben",
                "Summationsstelle verschieben",
                "Blöcke vertauschen",
                "Blöcke zusammenfassen",
                "Massons Regelungstheorie"
            ]
        )
        
        if rule_type == "Verzweigungspunkt verschieben":
            self._show_branch_point_rules()
        elif rule_type == "Summationsstelle verschieben":
            self._show_summing_point_rules()
        elif rule_type == "Blöcke vertauschen":
            self._show_block_interchange()
        elif rule_type == "Blöcke zusammenfassen":
            self._show_block_reduction()
        elif rule_type == "Massons Regelungstheorie":
            self._show_masons_rule()
    
    def _show_branch_point_rules(self):
        """Verzweigungspunkt-Regeln"""
        st.markdown("### Verzweigungspunkt verschieben")
        
        st.markdown("**Regel 1: Verzweigungspunkt nach rechts (hinter Block)**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Vorher:")
            st.text("""
            X(s) →→ [G(s)] → Y(s)
                 ↓
                Z(s)
            """)
        
        with col2:
            st.text("Nachher:")
            st.text("""
            X(s) → [G(s)] →→ Y(s)
                            ↓
                      [1/G(s)] → Z(s)
            """)
        
        st.markdown("**Regel 2: Verzweigungspunkt nach links (vor Block)**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Vorher:")
            st.text("""
            X(s) → [G(s)] →→ Y(s)
                            ↓
                           Z(s)
            """)
        
        with col2:
            st.text("Nachher:")
            st.text("""
            X(s) →→ [G(s)] → Y(s)
                 ↓
              [G(s)] → Z(s)
            """)
        
        st.info("💡 **Merkregel**: Beim Verschieben nach rechts: Division durch G(s), nach links: Multiplikation mit G(s)")
    
    def _show_summing_point_rules(self):
        """Summationsstellen-Regeln"""
        st.markdown("### Summationsstelle verschieben")
        
        st.markdown("**Regel 1: Summationsstelle nach rechts (hinter Block)**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Vorher:")
            st.text("""
            X₁(s) → + → [G(s)] → Y(s)
                    ↑
                   X₂(s)
            """)
        
        with col2:
            st.text("Nachher:")
            st.text("""
            X₁(s) → [G(s)] → + → Y(s)
                              ↑
                    X₂(s) → [G(s)]
            """)
        
        st.markdown("**Regel 2: Summationsstelle nach links (vor Block)**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Vorher:")
            st.text("""
            X₁(s) → [G(s)] → + → Y(s)
                              ↑
                             X₂(s)
            """)
        
        with col2:
            st.text("Nachher:")
            st.text("""
            X₁(s) → + → [G(s)] → Y(s)
                    ↑
          X₂(s) → [1/G(s)]
            """)
        
        st.info("💡 **Merkregel**: Beim Verschieben nach rechts: alle Eingänge durch G(s), nach links: zusätzliche Eingänge durch 1/G(s)")
    
    def _show_block_interchange(self):
        """Blöcke vertauschen"""
        st.markdown("### Blöcke vertauschen")
        
        st.markdown("**Reihenschaltung - Reihenfolge vertauschen:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Vorher:")
            st.text("X(s) → [G₁(s)] → [G₂(s)] → Y(s)")
        
        with col2:
            st.text("Nachher:")
            st.text("X(s) → [G₂(s)] → [G₁(s)] → Y(s)")
        
        st.latex(r"G_1(s) \cdot G_2(s) = G_2(s) \cdot G_1(s)")
        
        st.markdown("**Parallelschaltung - Reihenfolge vertauschen:**")
        st.text("""
        X(s) →→ [G₁(s)] →→ + → Y(s)
              ↓           ↑
              [G₂(s)] →→→→
        """)
        st.latex(r"G_1(s) + G_2(s) = G_2(s) + G_1(s)")
        
        st.info("💡 **Wichtig**: Vertauschen ist nur bei kommutativen Operationen (Multiplikation, Addition) möglich!")
    
    def _show_block_reduction(self):
        """Blöcke zusammenfassen"""
        st.markdown("### Blöcke zusammenfassen")
        
        st.markdown("**1. Mehrere Rückkopplungen:**")
        st.text("""
        X(s) → + → [G₁(s)] → + → [G₂(s)] → Y(s)
               ↑              ↑
               - ← [H₁(s)] ←← -
               ↑              ↑
               - ← [H₂(s)] ←←←←←←←←←←←←←
        """)
        
        st.markdown("**Schritt-für-Schritt-Reduktion:**")
        
        step = st.selectbox("Wählen Sie einen Reduktionsschritt:", 
                           ["Innere Schleife zuerst", "Äußere Schleife zuerst", "Direkte Berechnung"])
        
        if step == "Innere Schleife zuerst":
            st.markdown("**Schritt 1**: Innere Rückkopplung mit H₁(s)")
            st.latex(r"G_{12}(s) = \frac{G_1(s) \cdot G_2(s)}{1 + G_2(s) \cdot H_1(s)}")
            
            st.markdown("**Schritt 2**: Äußere Rückkopplung mit H₂(s)")
            st.latex(r"G_{ges}(s) = \frac{G_{12}(s)}{1 + G_{12}(s) \cdot H_2(s)}")
        
        elif step == "Äußere Schleife zuerst":
            st.markdown("**Schritt 1**: Äußere Rückkopplung mit H₂(s)")
            st.latex(r"G_{12}(s) = \frac{G_1(s) \cdot G_2(s)}{1 + G_1(s) \cdot G_2(s) \cdot H_2(s)}")
            
            st.markdown("**Schritt 2**: Innere Rückkopplung mit H₁(s)")
            st.latex(r"G_{ges}(s) = \frac{G_{12}(s)}{1 + G_2(s) \cdot H_1(s) \cdot \frac{1}{1 + G_1(s) \cdot G_2(s) \cdot H_2(s)}}")
        
        elif step == "Direkte Berechnung":
            st.markdown("**Direkte Formel für mehrfache Rückkopplung:**")
            st.latex(r"G_{ges}(s) = \frac{G_1(s) \cdot G_2(s)}{1 + G_2(s) \cdot H_1(s) + G_1(s) \cdot G_2(s) \cdot H_2(s)}")
    
    def _show_masons_rule(self):
        """Massons Regelungstheorie"""
        st.markdown("### Massons Regelungstheorie (Signal-Fluss-Diagramme)")
        
        st.markdown("""
        **Massons Gewinnformel:**
        """)
        st.latex(r"G = \frac{1}{\Delta} \sum_{k} P_k \Delta_k")
        
        st.markdown("""
        **Dabei bedeuten:**
        - $G$: Gesamtübertragungsfunktion
        - $P_k$: Verstärkung des k-ten Vorwärtspfades
        - $\Delta$: Determinante des Systems
        - $\Delta_k$: Determinante ohne Schleifen, die den k-ten Pfad berühren
        """)
        
        st.markdown("**Berechnung der Determinante:**")
        st.latex(r"\Delta = 1 - \sum L_i + \sum L_i L_j - \sum L_i L_j L_k + \ldots")
        
        st.markdown("""
        **Dabei:**
        - $L_i$: Schleifenverstärkung der i-ten Schleife
        - $L_i L_j$: Produkt nicht-berührender Schleifen
        - Vorzeichen wechseln pro Ordnung
        """)
        
        st.markdown("**Beispiel: Einfacher Regelkreis**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("""
            Signal-Fluss-Diagramm:
            
            W → [G₀] → Y
                ↓     ↑
                [H] ←←
            """)
        
        with col2:
            st.markdown("**Berechnung:**")
            st.markdown("- Vorwärtspfad: $P_1 = G_0$")
            st.markdown("- Schleife: $L_1 = -G_0 H$")
            st.markdown("- Determinante: $\\Delta = 1 - (-G_0 H) = 1 + G_0 H$")
            st.markdown("- $\\Delta_1 = 1$ (keine unberührte Schleife)")
            
            st.latex(r"G = \frac{P_1 \Delta_1}{\Delta} = \frac{G_0 \cdot 1}{1 + G_0 H}")
        
        st.info("💡 **Vorteil**: Massons Regel ist besonders nützlich bei komplexen Systemen mit vielen Schleifen und Pfaden.")
    
    def interactive_calculator(self):
        """Interaktiver Blockschaltbild-Rechner"""
        st.subheader("Interaktiver Blockschaltbild-Rechner")
        
        calculation_type = st.selectbox(
            "Wählen Sie eine Berechnungsart:",
            [
                "Zwei Blöcke in Reihe",
                "Zwei Blöcke parallel",
                "Einfacher Regelkreis",
                "Regelkreis mit Störung",
                "Kaskadierte Regelung"
            ]
        )
        
        if calculation_type == "Zwei Blöcke in Reihe":
            self._calc_series()
        elif calculation_type == "Zwei Blöcke parallel":
            self._calc_parallel()
        elif calculation_type == "Einfacher Regelkreis":
            self._calc_simple_control()
        elif calculation_type == "Regelkreis mit Störung":
            self._calc_control_with_disturbance()
        elif calculation_type == "Kaskadierte Regelung":
            self._calc_cascade_control()
    
    def _calc_series(self):
        """Berechnung Reihenschaltung"""
        st.markdown("### Reihenschaltung von zwei Blöcken")
        
        col1, col2 = st.columns(2)
        
        with col1:
            g1 = st.text_input("G₁(s) =", value="1/(s+1)", key="calc_series_g1")
            g2 = st.text_input("G₂(s) =", value="2/(s+2)", key="calc_series_g2")
        
        with col2:
            if st.button("Berechnen", key="calc_series_btn"):
                try:
                    s = sp.Symbol('s')
                    G1 = safe_sympify(g1)
                    G2 = safe_sympify(g2)
                    
                    G_total = sp.simplify(G1 * G2)
                    
                    display_step_by_step([
                        ("Gegeben", f"$G_1(s) = {sp.latex(G1)}$, $G_2(s) = {sp.latex(G2)}$"),
                        ("Formel", r"$G_{ges}(s) = G_1(s) \cdot G_2(s)$"),
                        ("Einsetzen", f"$G_{{ges}}(s) = {sp.latex(G1)} \\cdot {sp.latex(G2)}$"),
                        ("Ergebnis", f"$G_{{ges}}(s) = {sp.latex(G_total)}$")
                    ])
                    
                    # Analyse
                    st.markdown("**Analyse:**")
                    poles_g1 = sp.solve(sp.denom(G1), s)
                    poles_g2 = sp.solve(sp.denom(G2), s)
                    poles_total = sp.solve(sp.denom(G_total), s)
                    
                    st.write(f"Pole G₁(s): {poles_g1}")
                    st.write(f"Pole G₂(s): {poles_g2}")
                    st.write(f"Pole G_ges(s): {poles_total}")
                    
                except Exception as e:
                    st.error(f"Fehler: {e}")
    
    def _calc_parallel(self):
        """Berechnung Parallelschaltung"""
        st.markdown("### Parallelschaltung von zwei Blöcken")
        
        col1, col2 = st.columns(2)
        
        with col1:
            g1 = st.text_input("G₁(s) =", value="1/(s+1)", key="calc_parallel_g1")
            g2 = st.text_input("G₂(s) =", value="1/(s+2)", key="calc_parallel_g2")
        
        with col2:
            if st.button("Berechnen", key="calc_parallel_btn"):
                try:
                    s = sp.Symbol('s')
                    G1 = safe_sympify(g1)
                    G2 = safe_sympify(g2)
                    
                    G_total = sp.simplify(G1 + G2)
                    
                    display_step_by_step([
                        ("Gegeben", f"$G_1(s) = {sp.latex(G1)}$, $G_2(s) = {sp.latex(G2)}$"),
                        ("Formel", r"$G_{ges}(s) = G_1(s) + G_2(s)$"),
                        ("Einsetzen", f"$G_{{ges}}(s) = {sp.latex(G1)} + {sp.latex(G2)}$"),
                        ("Ergebnis", f"$G_{{ges}}(s) = {sp.latex(G_total)}$")
                    ])
                    
                except Exception as e:
                    st.error(f"Fehler: {e}")
    
    def _calc_simple_control(self):
        """Berechnung einfacher Regelkreis"""
        st.markdown("### Einfacher Regelkreis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            g0 = st.text_input("Strecke G₀(s) =", value="1/(s*(s+1))", key="ctrl_g0")
            gr = st.text_input("Regler Gr(s) =", value="2", key="ctrl_gr")
            h = st.text_input("Rückführung H(s) =", value="1", key="ctrl_h")
        
        with col2:
            if st.button("Berechnen", key="ctrl_btn"):
                try:
                    s = sp.Symbol('s')
                    G0 = safe_sympify(g0)
                    Gr = safe_sympify(gr)
                    H = safe_sympify(h)
                    
                    # Offene Kette
                    G_open = Gr * G0
                    
                    # Geschlossene Kette (Führungsverhalten)
                    G_w = sp.simplify(G_open / (1 + G_open * H))
                    
                    # Störverhalten (falls G0 die Störstrecke ist)
                    G_z = sp.simplify(G0 / (1 + G_open * H))
                    
                    display_step_by_step([
                        ("Gegeben", f"$G_0(s) = {sp.latex(G0)}$, $G_r(s) = {sp.latex(Gr)}$, $H(s) = {sp.latex(H)}$"),
                        ("Offene Kette", f"$G_0(s) = G_r(s) \\cdot G_0(s) = {sp.latex(G_open)}$"),
                        ("Führungsverhalten", f"$G_w(s) = \\frac{{G_0(s)}}{{1 + G_0(s) H(s)}} = {sp.latex(G_w)}$"),
                        ("Störverhalten", f"$G_z(s) = \\frac{{G_0(s)}}{{1 + G_0(s) H(s)}} = {sp.latex(G_z)}$")
                    ])
                    
                    # Stabilitätsanalyse
                    char_poly = 1 + G_open * H
                    poles = sp.solve(char_poly, s)
                    
                    st.markdown("**Stabilitätsanalyse:**")
                    st.write(f"Charakteristisches Polynom: {sp.simplify(char_poly)}")
                    st.write(f"Pole: {poles}")
                    
                    stable = all(sp.re(pole) < 0 for pole in poles if pole.is_real or pole.is_complex)
                    if stable:
                        st.success("✓ System ist stabil")
                    else:
                        st.error("✗ System ist instabil")
                    
                except Exception as e:
                    st.error(f"Fehler: {e}")
    
    def _calc_control_with_disturbance(self):
        """Regelkreis mit Störung"""
        st.markdown("### Regelkreis mit Störung")
        st.text("""
        Blockschaltbild:
        
        W(s) → + → [Gr(s)] → + → [G0(s)] → Y(s)
               ↑              ↑
               - ← [H(s)] ←←← Z(s) (Störung)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            gr = st.text_input("Regler Gr(s) =", value="10", key="dist_gr")
            g0 = st.text_input("Strecke G₀(s) =", value="1/(s+1)", key="dist_g0")
            h = st.text_input("Rückführung H(s) =", value="1", key="dist_h")
        
        with col2:
            if st.button("Berechnen", key="dist_btn"):
                try:
                    s = sp.Symbol('s')
                    Gr = safe_sympify(gr)
                    G0 = safe_sympify(g0)
                    H = safe_sympify(h)
                    
                    # Übertragungsfunktionen
                    G_open = Gr * G0
                    
                    # Führungsverhalten W(s) → Y(s)
                    G_w = sp.simplify(G_open / (1 + G_open * H))
                    
                    # Störverhalten Z(s) → Y(s)
                    G_z = sp.simplify(G0 / (1 + G_open * H))
                    
                    display_step_by_step([
                        ("Offene Kette", f"$G_0(s) = {sp.latex(G_open)}$"),
                        ("Führungsverhalten", f"$\\frac{{Y(s)}}{{W(s)}} = {sp.latex(G_w)}$"),
                        ("Störverhalten", f"$\\frac{{Y(s)}}{{Z(s)}} = {sp.latex(G_z)}$"),
                        ("Superposition", "$Y(s) = G_w(s) \\cdot W(s) + G_z(s) \\cdot Z(s)$")
                    ])
                    
                    # Stationäre Werte
                    st.markdown("**Stationäre Genauigkeit:**")
                    
                    # Führungssprung
                    G_w_dc = G_w.subs(s, 0)
                    st.write(f"Führungssprung: $G_w(0) = {G_w_dc}$")
                    
                    # Störsprung
                    G_z_dc = G_z.subs(s, 0)
                    st.write(f"Störsprung: $G_z(0) = {G_z_dc}$")
                    
                    if abs(G_w_dc - 1) < 0.01:
                        st.success("✓ Stationär genaue Führung")
                    else:
                        st.warning(f"⚠ Bleibende Führungsabweichung: {1 - G_w_dc}")
                    
                    if abs(G_z_dc) < 0.01:
                        st.success("✓ Störungen werden vollständig ausgeregelt")
                    else:
                        st.warning(f"⚠ Bleibende Störabweichung: {G_z_dc}")
                    
                except Exception as e:
                    st.error(f"Fehler: {e}")
    
    def _calc_cascade_control(self):
        """Kaskadierte Regelung"""
        st.markdown("### Kaskadierte Regelung")
        st.text("""
        Blockschaltbild:
        
        W → + → [Gr1] → + → [Gr2] → [G01] → + → [G02] → Y
            ↑           ↑                    ↑
            - ←←←←←←←←← - ←←←←←←←←←←←←←←←←← Z (Störung)
                        ↑
                        [H1] ←← (innere Rückführung)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            gr1 = st.text_input("Äußerer Regler Gr1(s) =", value="2", key="casc_gr1")
            gr2 = st.text_input("Innerer Regler Gr2(s) =", value="5", key="casc_gr2")
            g01 = st.text_input("Innere Strecke G01(s) =", value="1/(s+1)", key="casc_g01")
            g02 = st.text_input("Äußere Strecke G02(s) =", value="1/(s+2)", key="casc_g02")
            h1 = st.text_input("Innere Rückführung H1(s) =", value="1", key="casc_h1")
        
        with col2:
            if st.button("Berechnen", key="casc_btn"):
                try:
                    s = sp.Symbol('s')
                    Gr1 = safe_sympify(gr1)
                    Gr2 = safe_sympify(gr2)
                    G01 = safe_sympify(g01)
                    G02 = safe_sympify(g02)
                    H1 = safe_sympify(h1)
                    
                    # Innerer Kreis
                    G_inner = sp.simplify((Gr2 * G01) / (1 + Gr2 * G01 * H1))
                    
                    # Gesamtsystem
                    G_total = sp.simplify((Gr1 * G_inner * G02) / (1 + Gr1 * G_inner * G02))
                    
                    display_step_by_step([
                        ("Innerer Kreis", f"$G_i(s) = \\frac{{G_{{r2}}(s) G_{{01}}(s)}}{{1 + G_{{r2}}(s) G_{{01}}(s) H_1(s)}} = {sp.latex(G_inner)}$"),
                        ("Gesamtsystem", f"$G_{{ges}}(s) = \\frac{{G_{{r1}}(s) G_i(s) G_{{02}}(s)}}{{1 + G_{{r1}}(s) G_i(s) G_{{02}}(s)}} = {sp.latex(G_total)}$")
                    ])
                    
                    st.markdown("**Vorteile der Kaskadenregelung:**")
                    st.markdown("""
                    - Bessere Störunterdrückung
                    - Schnellere Reaktion auf Störungen
                    - Stabilisierung instabiler Teilsysteme
                    - Entkopplung von Störungen
                    """)
                    
                except Exception as e:
                    st.error(f"Fehler: {e}")
    
    def complex_structures(self):
        """Komplexe Strukturen"""
        st.subheader("Komplexe Regelungsstrukturen")
        
        structure_type = st.selectbox(
            "Wählen Sie eine komplexe Struktur:",
            [
                "Zwei-Freiheitsgrade-Regler",
                "Smith-Prädiktor",
                "Mehrgröße nsystem",
                "Entkopplung"
            ]
        )
        
        if structure_type == "Zwei-Freiheitsgrade-Regler":
            self._show_two_dof_controller()
        elif structure_type == "Smith-Prädiktor":
            self._show_smith_predictor()
        elif structure_type == "Mehrgrößensystem":
            self._show_mimo_system()
        elif structure_type == "Entkopplung":
            self._show_decoupling()
    
    def _show_two_dof_controller(self):
        """Zwei-Freiheitsgrade-Regler"""
        st.markdown("### Zwei-Freiheitsgrade-Regler")
        
        st.text("""
        Blockschaltbild:
        
        W(s) → [Gr1(s)] →→ + → [G0(s)] → Y(s)
                           ↑              ↓
              + ←← [Gr2(s)] ← - ←←←←←←←←← H(s)
              ↑
           [Gv(s)] ← W(s)
        """)
        
        st.markdown("""
        **Vorteile:**
        - Unabhängige Einstellung von Führungs- und Störverhalten
        - Gr1: Führungsverhalten optimieren
        - Gr2: Störverhalten und Stabilität optimieren
        - Gv: Vorfilter für Führungsglättung
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            gr1 = st.text_input("Führungsregler Gr1(s) =", value="1", key="2dof_gr1")
            gr2 = st.text_input("Störregler Gr2(s) =", value="10", key="2dof_gr2")
            gv = st.text_input("Vorfilter Gv(s) =", value="1/(0.1*s+1)", key="2dof_gv")
            g0 = st.text_input("Strecke G0(s) =", value="1/(s*(s+1))", key="2dof_g0")
        
        with col2:
            if st.button("Berechnen", key="2dof_btn"):
                try:
                    s = sp.Symbol('s')
                    Gr1 = safe_sympify(gr1)
                    Gr2 = safe_sympify(gr2)
                    Gv = safe_sympify(gv)
                    G0 = safe_sympify(g0)
                    
                    # Übertragungsfunktionen
                    denominator = 1 + Gr2 * G0
                    G_w = sp.simplify((Gv * Gr1 * G0) / denominator)
                    G_z = sp.simplify(G0 / denominator)
                    
                    st.latex(f"G_w(s) = {sp.latex(G_w)}")
                    st.latex(f"G_z(s) = {sp.latex(G_z)}")
                    
                    st.success("✓ Führungs- und Störverhalten können unabhängig optimiert werden!")
                    
                except Exception as e:
                    st.error(f"Fehler: {e}")
    
    def _show_smith_predictor(self):
        """Smith-Prädiktor"""
        st.markdown("### Smith-Prädiktor (für Totzeitstrecken)")
        
        st.text("""
        Blockschaltbild:
        
        W → + → [Gr] → + → [G0·e^(-Tt·s)] → Y
            ↑          ↑
            - ←←←←←←← + ←← [G0] (Modell ohne Totzeit)
                      ↓
                    [e^(-Tt·s)] (Totzeitmodell)
        """)
        
        st.markdown("""
        **Prinzip:**
        - Totzeitkompensation durch Parallelmodell
        - Gr wird für G0 (ohne Totzeit) ausgelegt
        - Robustheit gegenüber Modellunsicherheiten wichtig
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            gr = st.text_input("Regler Gr(s) =", value="2", key="smith_gr")
            g0 = st.text_input("Strecke ohne Totzeit G0(s) =", value="1/(s+1)", key="smith_g0")
            tt = st.number_input("Totzeit Tt [s]:", value=2.0, step=0.1, key="smith_tt")
        
        with col2:
            if st.button("Berechnen", key="smith_btn"):
                try:
                    s = sp.Symbol('s')
                    Gr = safe_sympify(gr)
                    G0 = safe_sympify(g0)
                    
                    # Vereinfachte Übertragungsfunktion (ideal)
                    G_w_ideal = sp.simplify((Gr * G0) / (1 + Gr * G0))
                    
                    st.markdown("**Ideale Übertragungsfunktion (perfektes Modell):**")
                    st.latex(f"G_w(s) = e^{{-T_t s}} \\cdot {sp.latex(G_w_ideal)}")
                    
                    st.markdown(f"**Totzeit:** $T_t = {tt}$ s")
                    
                    st.info("💡 Der Smith-Prädiktor eliminiert die Totzeit aus der charakteristischen Gleichung!")
                    
                    st.warning("⚠️ **Achtung**: Funktioniert nur bei genauem Totzeitmodell. Modellfehler können zu Instabilität führen.")
                    
                except Exception as e:
                    st.error(f"Fehler: {e}")
    
    def _show_mimo_system(self):
        """MIMO-System"""
        st.markdown("### Mehrgrößensystem (MIMO)")
        
        st.markdown("""
        **2×2 System:**
        """)
        st.latex(r"""
        \begin{bmatrix} Y_1(s) \\ Y_2(s) \end{bmatrix} = 
        \begin{bmatrix} G_{11}(s) & G_{12}(s) \\ G_{21}(s) & G_{22}(s) \end{bmatrix}
        \begin{bmatrix} U_1(s) \\ U_2(s) \end{bmatrix}
        """)
        
        st.markdown("**Probleme:**")
        st.markdown("""
        - Kopplung zwischen den Kanälen
        - G₁₂(s): Einfluss von U₁ auf Y₂  
        - G₂₁(s): Einfluss von U₂ auf Y₁
        - Komplexe Regelung erforderlich
        """)
        
        # Beispielrechnung
        col1, col2 = st.columns(2)
        
        with col1:
            g11 = st.text_input("G₁₁(s) =", value="2/(s+1)", key="mimo_g11")
            g12 = st.text_input("G₁₂(s) =", value="1/(s+2)", key="mimo_g12")
            g21 = st.text_input("G₂₁(s) =", value="0.5/(s+1)", key="mimo_g21")
            g22 = st.text_input("G₂₂(s) =", value="3/(s+3)", key="mimo_g22")
        
        with col2:
            if st.button("Analysieren", key="mimo_btn"):
                try:
                    s = sp.Symbol('s')
                    G11 = safe_sympify(g11)
                    G12 = safe_sympify(g12)
                    G21 = safe_sympify(g21)
                    G22 = safe_sympify(g22)
                    
                    # RGA (Relative Gain Array) bei s=0
                    G11_dc = G11.subs(s, 0)
                    G12_dc = G12.subs(s, 0)
                    G21_dc = G21.subs(s, 0)
                    G22_dc = G22.subs(s, 0)
                    
                    # Korrekte RGA Berechnung: Λ = G .* (G^(-1))^T
                    G_matrix = np.array([[float(G11_dc), float(G12_dc)], 
                                       [float(G21_dc), float(G22_dc)]])
                    
                    try:
                        G_inv = np.linalg.inv(G_matrix)
                        G_inv_T = G_inv.T
                        
                        # Element-weise Multiplikation
                        Lambda = G_matrix * G_inv_T
                        
                        lambda_11 = Lambda[0, 0]
                        lambda_12 = Lambda[0, 1]
                        lambda_21 = Lambda[1, 0] 
                        lambda_22 = Lambda[1, 1]
                        
                        st.markdown("**Relative Gain Array (RGA) bei s=0:**")
                        st.latex(f"\\Lambda = \\begin{{bmatrix}} {lambda_11:.3f} & {lambda_12:.3f} \\\\ {lambda_21:.3f} & {lambda_22:.3f} \\end{{bmatrix}}")
                        
                        # RGA-Eigenschaften prüfen
                        row_sum = lambda_11 + lambda_12
                        col_sum = lambda_11 + lambda_21
                        
                        st.write(f"Zeilensumme: {row_sum:.6f}")
                        st.write(f"Spaltensumme: {col_sum:.6f}")
                        
                        if abs(lambda_11 - 1) < 0.1:
                            st.success("✓ Schwache Kopplung - diagonale Regelung möglich")
                        elif lambda_11 < 0:
                            st.error("✗ Negative RGA-Elemente - System schwer regelbar")
                        else:
                            st.warning("⚠ Starke Kopplung - Entkopplungsregelung empfohlen")
                            
                    except np.linalg.LinAlgError:
                        st.error("❌ Matrix ist singulär - RGA nicht berechenbar")
                    
                        st.markdown("**Relative Gain Array (RGA) bei s=0:**")
                        st.latex(f"\\Lambda = \\begin{{bmatrix}} {lambda_11:.3f} & {lambda_12:.3f} \\\\ {lambda_21:.3f} & {lambda_22:.3f} \\end{{bmatrix}}")
                        
                        # RGA-Eigenschaften prüfen
                        row_sum = lambda_11 + lambda_12
                        col_sum = lambda_11 + lambda_21
                        
                        st.write(f"Zeilensumme: {row_sum:.6f}")
                        st.write(f"Spaltensumme: {col_sum:.6f}")
                        
                        if abs(lambda_11 - 1) < 0.1:
                            st.success("✓ Schwache Kopplung - diagonale Regelung möglich")
                        elif lambda_11 < 0:
                            st.error("✗ Negative RGA-Elemente - System schwer regelbar")
                        else:
                            st.warning("⚠ Starke Kopplung - Entkopplungsregelung empfohlen")
                            
                    except np.linalg.LinAlgError:
                        st.error("❌ Matrix ist singulär - RGA nicht berechenbar")
                    
                except Exception as e:
                    st.error(f"Fehler: {e}")
    
    def _show_decoupling(self):
        """Entkopplung"""
        st.markdown("### Entkopplungsregelung")
        
        st.markdown("""
        **Ziel:** Transformation des MIMO-Systems in mehrere SISO-Systeme
        """)
        
        st.markdown("**Dezentrale Entkopplung:**")
        st.latex(r"""
        D(s) = \begin{bmatrix} D_{11}(s) & D_{12}(s) \\ D_{21}(s) & D_{22}(s) \end{bmatrix}
        """)
        
        st.markdown("**Bedingung für vollständige Entkopplung:**")
        st.latex(r"G(s) \cdot D(s) = \text{diagonal}")
        
        st.markdown("**Statische Entkopplung (s=0):**")
        st.latex(r"D = G^{-1}(0) \cdot \text{diag}(G(0))")
        
        # Beispielrechnung
        if st.button("Beispiel berechnen", key="decoup_example"):
            st.markdown("**Beispiel:**")
            st.latex(r"G(0) = \begin{bmatrix} 2 & 1 \\ 0.5 & 3 \end{bmatrix}")
            
            # Determinante
            det_g = 2*3 - 1*0.5
            st.latex(f"\\det(G(0)) = {det_g}")
            
            # Inverse
            st.latex(r"G^{-1}(0) = \frac{1}{5.5} \begin{bmatrix} 3 & -1 \\ -0.5 & 2 \end{bmatrix}")
            
            # Entkopplungsmatrix
            st.latex(r"D = \begin{bmatrix} 3/5.5 & 0 \\ 0 & 3/5.5 \end{bmatrix} \cdot \begin{bmatrix} 3 & -1 \\ -0.5 & 2 \end{bmatrix}")
            
            st.success("✓ Nach Entkopplung: Zwei unabhängige SISO-Systeme mit Verstärkung 2 und 3")
