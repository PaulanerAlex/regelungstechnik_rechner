"""
Umfassende Übertragungsfunktions-Analyse
Kombiniert alle Analyse-Tools in einem Modul mit zentraler Eingabeverwaltung
"""

import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_module import BaseModule

try:
    from ..utils.display_utils import display_step_by_step, display_latex
except ImportError:
    def display_step_by_step(steps):
        """Fallback function for displaying steps"""
        for step_name, step_content in steps:
            st.markdown(f"**{step_name}:**")
            st.latex(step_content)
    
    def display_latex(content):
        """Fallback function for displaying LaTeX"""
        st.latex(content)

from ..utils.safe_sympify import safe_sympify

class TransferFunctionAnalysisModule(BaseModule):
    """Umfassende Übertragungsfunktions-Analyse mit zentraler Eingabeverwaltung"""
    
    def __init__(self):
        super().__init__(
            "🎯 Übertragungsfunktions-Analyse",
            "Umfassende Analyse von Übertragungsfunktionen: Pole-Nullstellen, Frequenzgang, Nyquist, Wurzelortskurve, Stabilität"
        )
    
    def render(self):
        """Hauptrender-Methode mit Tab-basierter Organisation"""
        self.display_description()
        
        # Zentrale Eingabe für Übertragungsfunktion
        self._transfer_function_input()
        
        # Tab-basierte Analyse-Organisation
        if 'tf_parsed' in st.session_state and st.session_state.tf_parsed is not None:
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🎯 Komplettanalyse",
                "📊 Pol-Nullstellen", 
                "📈 Frequenzgang",
                "🔄 Nyquist",
                "🌿 Wurzelortskurve",
                "⚖️ Stabilität"
            ])
            
            with tab1:
                self.complete_analysis()
            
            with tab2:
                self.pole_zero_analysis()
            
            with tab3:
                self.frequency_response_analysis()
            
            with tab4:
                self.nyquist_analysis()
            
            with tab5:
                self.root_locus_analysis()
            
            with tab6:
                self.stability_analysis()
    
    def _transfer_function_input(self):
        """Zentrale Eingabe für Übertragungsfunktionen mit Session State Management"""
        st.markdown("### 🎛️ Übertragungsfunktion definieren")
        
        # Eingabe-Container
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Zähler und Nenner getrennt eingeben
                numerator = st.text_input(
                    "Zähler (Numerator):",
                    value=st.session_state.get('tf_numerator', '10'),
                    help="Beispiele: 1, s+1, s**2+2*s+1 (automatische Multiplikation: 2s = 2*s)",
                    key='tf_numerator'
                )
                
                denominator = st.text_input(
                    "Nenner (Denominator):",
                    value=st.session_state.get('tf_denominator', 's**2+3*s+2'),
                    help="Beispiele: s+1, (s+1)*(s+2), s**2+2*s+1",
                    key='tf_denominator'
                )
            
            with col2:
                st.markdown("**Einstellungen:**")
                auto_simplify = st.checkbox("Automatisch vereinfachen", value=True)
                show_steps = st.checkbox("Parsing-Schritte zeigen", value=False)
        
        # Parse-Button
        if st.button("🔄 Übertragungsfunktion verarbeiten", type="primary"):
            self._parse_transfer_function(numerator, denominator, auto_simplify, show_steps)
        
        # Aktuelle Übertragungsfunktion anzeigen
        if 'tf_parsed' in st.session_state and st.session_state.tf_parsed is not None:
            st.success("✅ Übertragungsfunktion erfolgreich geladen")
            
            # Anzeige der verarbeiteten Übertragungsfunktion
            with st.expander("📋 Aktuelle Übertragungsfunktion anzeigen"):
                G_s = st.session_state.tf_parsed['G_s']
                st.latex(f"G(s) = {sp.latex(G_s)}")
                
                # Zusätzliche Informationen
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    num_deg = sp.degree(sp.numer(G_s), st.session_state.tf_symbols['s'])
                    st.metric("Zählergrad", num_deg)
                
                with col_info2:
                    den_deg = sp.degree(sp.denom(G_s), st.session_state.tf_symbols['s'])
                    st.metric("Nennergrad", den_deg)
                
                with col_info3:
                    system_type = "Eigentlich" if num_deg <= den_deg else "Uneigentlich"
                    st.metric("Systemtyp", system_type)
    
    def _parse_transfer_function(self, numerator, denominator, auto_simplify, show_steps):
        """Parse und validiere die Übertragungsfunktion"""
        try:
            # SymPy Symbole definieren
            s, omega = sp.symbols('s omega', real=True)
            symbols_dict = {'s': s, 'omega': omega, 'j': sp.I}
            
            if show_steps:
                st.markdown("#### 🔍 Parsing-Schritte")
                st.info(f"Eingabe: Zähler='{numerator}', Nenner='{denominator}'")
            
            # Parse Zähler und Nenner
            num_expr = safe_sympify(numerator, symbols_dict)
            den_expr = safe_sympify(denominator, symbols_dict)
            
            if show_steps:
                st.success(f"✅ Parsing erfolgreich:")
                st.write(f"Zähler: {num_expr}")
                st.write(f"Nenner: {den_expr}")
            
            # Übertragungsfunktion erstellen
            G_s = num_expr / den_expr
            
            # Vereinfachen falls gewünscht
            if auto_simplify:
                G_s = sp.simplify(G_s)
                if show_steps:
                    st.write(f"Vereinfacht: {G_s}")
            
            # In Session State speichern
            st.session_state.tf_parsed = {
                'G_s': G_s,
                'numerator': num_expr,
                'denominator': den_expr,
                'original_num_str': numerator,
                'original_den_str': denominator
            }
            
            st.session_state.tf_symbols = {
                's': s,
                'omega': omega,
                'j': sp.I
            }
            
            # Erfolgsbestätigung
            st.success("✅ Übertragungsfunktion erfolgreich verarbeitet!")
            
        except Exception as e:
            st.error(f"❌ Fehler beim Verarbeiten der Übertragungsfunktion: {e}")
            st.info("💡 **Eingabe-Tipps:**")
            st.markdown("""
            - Verwenden Sie `**` für Potenzen: `s**2` statt `s^2`
            - Multiplikation wird automatisch erkannt: `2s` → `2*s`
            - Klammern für Gruppierung: `(s+1)*(s+2)`
            - Komplexe Zahlen: `j` für imaginäre Einheit
            """)
    
    def complete_analysis(self):
        """Komplette Analyse mit auswählbaren Komponenten"""
        st.subheader("🎯 Komplettanalyse")
        
        if st.session_state.tf_parsed is None:
            st.warning("⚠️ Bitte definieren Sie zunächst eine Übertragungsfunktion.")
            return
        
        st.markdown("Wählen Sie die gewünschten Analysekomponenten:")
        
        # Auswahl der Analysekomponenten
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_properties = st.checkbox("📋 Systemeigenschaften", value=True)
            show_poles_zeros = st.checkbox("📊 Pol-Nullstellen-Diagramm", value=True)
        
        with col2:
            show_frequency = st.checkbox("📈 Frequenzgang (Bode)", value=True)
            show_nyquist = st.checkbox("🔄 Nyquist-Diagramm", value=True)
        
        with col3:
            show_stability = st.checkbox("⚖️ Stabilitätsanalyse", value=True)
            show_margins = st.checkbox("📏 Stabilitätsreserven", value=True)
        
        if st.button("🚀 Komplettanalyse starten", type="primary"):
            
            G_s = st.session_state.tf_parsed['G_s']
            s = st.session_state.tf_symbols['s']
            omega = st.session_state.tf_symbols['omega']
            
            # Systemeigenschaften
            if show_properties:
                self._show_system_properties(G_s, s)
            
            # Pol-Nullstellen-Diagramm
            if show_poles_zeros:
                st.markdown("---")
                st.markdown("### 📍 Pol-Nullstellen-Diagramm")
                self._create_pole_zero_plot(G_s, s)
            
            # Frequenzgang
            if show_frequency:
                st.markdown("---")
                st.markdown("### 📈 Bode-Diagramm")
                self._create_bode_plot(G_s, s, omega)
            
            # Nyquist
            if show_nyquist:
                st.markdown("---")
                st.markdown("### 🔄 Nyquist-Diagramm")
                self._create_nyquist_plot(G_s, s, omega)
            
            # Stabilitätsanalyse
            if show_stability:
                st.markdown("---")
                st.markdown("### ⚖️ Stabilitätsanalyse")
                self._analyze_stability(G_s, s)
            
            # Stabilitätsreserven
            if show_margins:
                st.markdown("---")
                st.markdown("### 📏 Stabilitätsreserven")
                self._calculate_stability_margins(G_s, s, omega)
    
    def _show_system_properties(self, G_s, s):
        """Zeige grundlegende Systemeigenschaften"""
        st.markdown("### 📋 Systemeigenschaften")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Übertragungsfunktion:**")
            st.latex(f"G(s) = {sp.latex(G_s)}")
            
            # Systemordnung
            num_degree = sp.degree(sp.numer(G_s), s)
            den_degree = sp.degree(sp.denom(G_s), s)
            
            st.markdown("**Systemparameter:**")
            st.write(f"• Zählergrad: {num_degree}")
            st.write(f"• Nennergrad: {den_degree}")
            st.write(f"• Systemordnung: {den_degree}")
            st.write(f"• Systemtyp: {'Eigentlich' if num_degree <= den_degree else 'Uneigentlich'}")
        
        with col2:
            # Pole und Nullstellen
            poles = sp.solve(sp.denom(G_s), s)
            zeros = sp.solve(sp.numer(G_s), s)
            
            st.markdown("**Pole:**")
            if poles:
                for i, pole in enumerate(poles):
                    pole_val = complex(pole.evalf())
                    st.write(f"p{i+1} = {pole_val:.4f}")
            else:
                st.write("Keine Pole")
            
            st.markdown("**Nullstellen:**")
            if zeros:
                for i, zero in enumerate(zeros):
                    zero_val = complex(zero.evalf())
                    st.write(f"z{i+1} = {zero_val:.4f}")
            else:
                st.write("Keine endlichen Nullstellen")
            
            # DC-Verstärkung
            try:
                dc_gain = float(G_s.subs(s, 0))
                st.markdown("**DC-Verstärkung:**")
                st.write(f"K₀ = {dc_gain:.4f}")
                st.write(f"K₀ [dB] = {20*np.log10(abs(dc_gain)):.1f} dB")
            except:
                st.markdown("**DC-Verstärkung:**")
                st.write("Nicht definiert (Pol bei s=0)")
    
    def pole_zero_analysis(self):
        """Detaillierte Pol-Nullstellen-Analyse"""
        st.subheader("📊 Pol-Nullstellen-Analyse")
        
        if st.session_state.tf_parsed is None:
            st.warning("⚠️ Bitte definieren Sie zunächst eine Übertragungsfunktion.")
            return
        
        G_s = st.session_state.tf_parsed['G_s']
        s = st.session_state.tf_symbols['s']
        
        try:
            # Numerator und Denominator extrahieren
            num = sp.numer(G_s)
            den = sp.denom(G_s)
            
            # Pole und Nullstellen berechnen
            poles = sp.solve(den, s)
            zeros = sp.solve(num, s)
            
            # Anzeige der Ergebnisse in Spalten
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Pole")
                if poles:
                    for i, pole in enumerate(poles):
                        pole_val = complex(pole.evalf())
                        real_part = pole_val.real
                        imag_part = pole_val.imag
                        
                        # Pol-Typ bestimmen
                        if abs(imag_part) < 1e-10:  # Reeller Pol
                            if real_part < 0:
                                st.write(f"**p{i+1}** = {real_part:.3f} (stabil, reell)")
                                # Zeitkonstante
                                tau = -1/real_part
                                st.write(f"   → Zeitkonstante: τ = {tau:.3f} s")
                            else:
                                st.write(f"**p{i+1}** = {real_part:.3f} ⚠️ (instabil, reell)")
                        else:  # Komplexer Pol
                            magnitude = abs(pole_val)
                            angle = np.angle(pole_val) * 180 / np.pi
                            
                            if real_part < 0:
                                st.write(f"**p{i+1}** = {real_part:.3f} ± {abs(imag_part):.3f}j")
                                st.write(f"   → |p| = {magnitude:.3f}, ∠p = {angle:.1f}°")
                                
                                # Dämpfung und Eigenfrequenz
                                omega_n = magnitude
                                zeta = -real_part / omega_n
                                st.write(f"   → ωₙ = {omega_n:.3f} rad/s, ζ = {zeta:.3f}")
                                
                                if zeta < 1:
                                    omega_d = omega_n * np.sqrt(1 - zeta**2)
                                    st.write(f"   → ωd = {omega_d:.3f} rad/s (gedämpft)")
                                    
                                    # Einschwingzeit und Überschwingen
                                    t_s = 4 / (zeta * omega_n)  # 2% Einschwingzeit
                                    overshoot = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100
                                    st.write(f"   → ts ≈ {t_s:.2f} s, Überschwingen ≈ {overshoot:.1f}%")
                            else:
                                st.write(f"**p{i+1}** = {real_part:.3f} ± {abs(imag_part):.3f}j ⚠️ (instabil)")
                else:
                    st.write("Keine Pole gefunden")
            
            with col2:
                st.markdown("#### 🎯 Nullstellen")
                if zeros:
                    for i, zero in enumerate(zeros):
                        zero_val = complex(zero.evalf())
                        real_part = zero_val.real
                        imag_part = zero_val.imag
                        
                        if abs(imag_part) < 1e-10:  # Reelle Nullstelle
                            st.write(f"**z{i+1}** = {real_part:.3f} (reell)")
                            if real_part > 0:
                                st.write(f"   → Nichtminimalphasensystem!")
                        else:  # Komplexe Nullstelle
                            magnitude = abs(zero_val)
                            angle = np.angle(zero_val) * 180 / np.pi
                            st.write(f"**z{i+1}** = {real_part:.3f} ± {abs(imag_part):.3f}j")
                            st.write(f"   → |z| = {magnitude:.3f}, ∠z = {angle:.1f}°")
                            
                            if real_part > 0:
                                st.write(f"   → Nichtminimalphasensystem!")
                else:
                    st.write("Keine endlichen Nullstellen")
            
            # Systemanalyse
            st.markdown("#### 📈 Systemcharakteristik")
            
            # Stabilität
            stable_poles = all(complex(pole.evalf()).real < 0 for pole in poles)
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if stable_poles:
                    st.success("✅ **System ist stabil**")
                    st.write("Alle Pole in linker Halbebene")
                else:
                    st.error("❌ **System ist instabil**")
                    st.write("Pole in rechter Halbebene vorhanden")
            
            with col_b:
                # Systemordnung
                system_order = len(poles)
                st.info(f"📊 **Systemordnung: {system_order}**")
                
                # Systemtyp
                num_degree = sp.degree(num, s) if sp.degree(num, s) is not None else 0
                den_degree = sp.degree(den, s) if sp.degree(den, s) is not None else 0
                
                if num_degree <= den_degree:
                    st.write("Eigentliches System")
                else:
                    st.write("Uneigentliches System")
            
            with col_c:
                # Minimalphasensystem?
                if zeros:
                    min_phase = all(complex(zero.evalf()).real < 0 for zero in zeros)
                    if min_phase:
                        st.success("✅ **Minimalphasensystem**")
                        st.write("Alle Nullstellen in linker Halbebene")
                    else:
                        st.warning("⚠️ **Nichtminimalphasensystem**")
                        st.write("Nullstellen in rechter Halbebene")
                else:
                    st.success("✅ **Minimalphasensystem**")
                    st.write("Keine Nullstellen in rechter Halbebene")
            
            # Pol-Nullstellen-Diagramm
            self._create_pole_zero_plot(G_s, s)
            
            # Zusätzliche Analyse
            if st.checkbox("🔬 Erweiterte Analyse anzeigen"):
                self._extended_pole_zero_analysis(poles, zeros, G_s, s)
                
        except Exception as e:
            st.error(f"Fehler bei der Pol-Nullstellen-Analyse: {e}")
            import traceback
            with st.expander("Debug-Informationen"):
                st.code(traceback.format_exc())
    
    def _create_pole_zero_plot(self, G_s, s):
        """Erstelle Pol-Nullstellen-Diagramm"""
        try:
            # Numerator und Denominator extrahieren
            num = sp.numer(G_s)
            den = sp.denom(G_s)
            
            # Pole und Nullstellen berechnen
            poles = sp.solve(den, s)
            zeros = sp.solve(num, s)
            
            # Plot erstellen
            fig = go.Figure()
            
            # Pole hinzufügen (X-Symbole)
            if poles:
                pole_real = [complex(pole.evalf()).real for pole in poles]
                pole_imag = [complex(pole.evalf()).imag for pole in poles]
                
                fig.add_trace(go.Scatter(
                    x=pole_real, y=pole_imag,
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color='red',
                        line=dict(width=3)
                    ),
                    name='Pole',
                    hovertemplate='Pol<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>'
                ))
            
            # Nullstellen hinzufügen (O-Symbole)
            if zeros:
                zero_real = [complex(zero.evalf()).real for zero in zeros]
                zero_imag = [complex(zero.evalf()).imag for zero in zeros]
                
                fig.add_trace(go.Scatter(
                    x=zero_real, y=zero_imag,
                    mode='markers',
                    marker=dict(
                        symbol='circle-open',
                        size=12,
                        color='blue',
                        line=dict(width=3)
                    ),
                    name='Nullstellen',
                    hovertemplate='Nullstelle<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>'
                ))
            
            # Koordinatenachsen
            fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
            fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
            
            # Stabilität-Region (linke Halbebene) hervorheben
            all_reals = []
            all_imags = []
            
            if poles:
                pole_real = [complex(pole.evalf()).real for pole in poles]
                pole_imag = [complex(pole.evalf()).imag for pole in poles]
                all_reals.extend(pole_real)
                all_imags.extend(pole_imag)
            
            if zeros:
                zero_real = [complex(zero.evalf()).real for zero in zeros]
                zero_imag = [complex(zero.evalf()).imag for zero in zeros]
                all_reals.extend(zero_real)
                all_imags.extend(zero_imag)
            
            if all_reals:
                x_min = min(all_reals) - 1
                x_max = max(all_reals) + 1
                y_min = min(all_imags) - 1
                y_max = max(all_imags) + 1
                
                # Linke Halbebene (stabile Region) einfärben
                fig.add_shape(
                    type="rect",
                    x0=x_min, y0=y_min,
                    x1=0, y1=y_max,
                    fillcolor="lightgreen",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
                
                # Rechte Halbebene (instabile Region) einfärben
                fig.add_shape(
                    type="rect",
                    x0=0, y0=y_min,
                    x1=x_max, y1=y_max,
                    fillcolor="lightcoral",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
            else:
                x_min, x_max = -5, 5
                y_min, y_max = -5, 5
            
            # Legende hinzufügen
            fig.add_annotation(
                x=x_min + 0.1, y=y_max - 0.3,
                text="Stabile Region",
                showarrow=False,
                font=dict(color="darkgreen", size=12),
                bgcolor="lightgreen",
                opacity=0.8
            )
            
            if x_max > 0:
                fig.add_annotation(
                    x=x_max - 0.1, y=y_max - 0.3,
                    text="Instabile Region",
                    showarrow=False,
                    font=dict(color="darkred", size=12),
                    bgcolor="lightcoral",
                    opacity=0.8
                )
            
            fig.update_layout(
                title="Pol-Nullstellen-Diagramm",
                xaxis_title="Realteil",
                yaxis_title="Imaginärteil",
                showlegend=True,
                width=700, height=500,
                xaxis=dict(
                    showgrid=True,
                    zeroline=True,
                    range=[x_min, x_max]
                ),
                yaxis=dict(
                    showgrid=True,
                    zeroline=True,
                    range=[y_min, y_max],
                    scaleanchor="x",
                    scaleratio=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Fehler beim Erstellen des Pol-Nullstellen-Diagramms: {e}")
    
    def _extended_pole_zero_analysis(self, poles, zeros, G_s, s):
        """Erweiterte Pol-Nullstellen-Analyse"""
        st.markdown("#### 🔬 Erweiterte Analyse")
        
        # Komplex konjugierte Paare identifizieren
        st.markdown("**Komplex konjugierte Paare:**")
        
        complex_poles = []
        real_poles = []
        
        for pole in poles:
            pole_val = complex(pole.evalf())
            if abs(pole_val.imag) > 1e-10:
                # Prüfe ob bereits als konjugiertes Paar hinzugefügt
                already_added = False
                for existing in complex_poles:
                    if abs(existing.real - pole_val.real) < 1e-10 and abs(abs(existing.imag) - abs(pole_val.imag)) < 1e-10:
                        already_added = True
                        break
                
                if not already_added:
                    complex_poles.append(pole_val)
            else:
                real_poles.append(pole_val.real)
        
        col_ext1, col_ext2 = st.columns(2)
        
        with col_ext1:
            st.markdown("**Reelle Pole:**")
            for i, pole in enumerate(real_poles):
                st.write(f"p{i+1} = {pole:.4f}")
                if pole < 0:
                    tau = -1/pole
                    st.write(f"   → τ = {tau:.3f} s")
                    
        with col_ext2:
            st.markdown("**Komplexe Polpaare:**")
            for i, pole in enumerate(complex_poles):
                omega_n = abs(pole)
                zeta = -pole.real / omega_n
                st.write(f"p{i+1},p{i+2} = {pole.real:.3f} ± {abs(pole.imag):.3f}j")
                st.write(f"   → ωₙ = {omega_n:.3f} rad/s")
                st.write(f"   → ζ = {zeta:.3f}")
                
                if zeta < 1:
                    omega_d = omega_n * np.sqrt(1 - zeta**2)
                    st.write(f"   → ωd = {omega_d:.3f} rad/s")
        
        # Dominante Pole identifizieren
        st.markdown("**Dominante Pole:**")
        if poles:
            # Pole nach Entfernung zur imaginären Achse sortieren
            pole_distances = []
            for pole in poles:
                pole_val = complex(pole.evalf())
                if pole_val.real < 0:  # Nur stabile Pole betrachten
                    distance = abs(pole_val.real)
                    pole_distances.append((distance, pole_val, pole))
            
            if pole_distances:
                pole_distances.sort(key=lambda x: x[0])
                dominant_pole = pole_distances[0]
                
                st.write(f"Dominanter Pol: {dominant_pole[1]:.4f}")
                st.write(f"Entfernung zur jω-Achse: {dominant_pole[0]:.4f}")
                
                # Zeitkonstante des dominanten Pols
                if abs(dominant_pole[1].imag) < 1e-10:
                    tau_dom = 1/dominant_pole[0]
                    st.write(f"Dominante Zeitkonstante: τ = {tau_dom:.3f} s")
                else:
                    omega_n = abs(dominant_pole[1])
                    zeta = -dominant_pole[1].real / omega_n
                    tau_dom = 1/(zeta * omega_n)
                    st.write(f"Dominante Zeitkonstante: τ ≈ {tau_dom:.3f} s")
        
        # Pol-Nullstellen-Kopplung
        if zeros and poles:
            st.markdown("**Pol-Nullstellen-Dipole:**")
            
            # Finde nahe Pol-Nullstellen-Paare (Dipole)
            dipoles = []
            for zero in zeros:
                zero_val = complex(zero.evalf())
                for pole in poles:
                    pole_val = complex(pole.evalf())
                    distance = abs(zero_val - pole_val)
                    if distance < 0.1 * max(abs(zero_val), abs(pole_val), 1):  # Relative Nähe
                        dipoles.append((zero_val, pole_val, distance))
            
            if dipoles:
                for i, (zero, pole, dist) in enumerate(dipoles):
                    st.write(f"Dipol {i+1}: z={zero:.3f}, p={pole:.3f} (Abstand: {dist:.4f})")
                    st.write(f"   → Einfluss auf Übertragungsverhalten minimal")
            else:
                st.write("Keine signifikanten Pol-Nullstellen-Dipole gefunden")
    
    def frequency_response_analysis(self):
        """Detaillierte Frequenzgang-Analyse"""
        st.subheader("📈 Frequenzgang-Analyse")
        
        if st.session_state.tf_parsed is None:
            st.warning("⚠️ Bitte definieren Sie zunächst eine Übertragungsfunktion.")
            return
        
        G_s = st.session_state.tf_parsed['G_s']
        s = st.session_state.tf_symbols['s']
        omega = st.session_state.tf_symbols['omega']
        
        # Placeholder für Frequenzgang-Analyse
        st.info("🚧 Frequenzgang-Analyse wird in Schritt 3 implementiert...")
        self._create_bode_plot(G_s, s, omega)
    
    def nyquist_analysis(self):
        """Detaillierte Nyquist-Analyse"""
        st.subheader("🔄 Nyquist-Analyse")
        
        if st.session_state.tf_parsed is None:
            st.warning("⚠️ Bitte definieren Sie zunächst eine Übertragungsfunktion.")
            return
        
        # Placeholder für Nyquist-Analyse
        st.info("🚧 Nyquist-Analyse wird in Schritt 4 implementiert...")
    
    def root_locus_analysis(self):
        """Detaillierte Wurzelortskurven-Analyse"""
        st.subheader("🌿 Wurzelortskurven-Analyse")
        
        if st.session_state.tf_parsed is None:
            st.warning("⚠️ Bitte definieren Sie zunächst eine Übertragungsfunktion.")
            return
        
        # Placeholder für Wurzelortskurven-Analyse
        st.info("🚧 Wurzelortskurven-Analyse wird in Schritt 4 implementiert...")
    
    def stability_analysis(self):
        """Detaillierte Stabilitätsanalyse"""
        st.subheader("⚖️ Stabilitätsanalyse")
        
        if st.session_state.tf_parsed is None:
            st.warning("⚠️ Bitte definieren Sie zunächst eine Übertragungsfunktion.")
            return
        
        # Placeholder für Stabilitätsanalyse
        st.info("🚧 Stabilitätsanalyse wird in Schritt 5 implementiert...")
    
    # Placeholder-Methoden für die Komplettanalyse
    def _create_bode_plot(self, G_s, s, omega):
        """Erstelle Bode-Diagramm"""
        st.info("🚧 Bode-Diagramm wird in Schritt 3 implementiert...")
    
    def _create_nyquist_plot(self, G_s, s, omega):
        """Erstelle Nyquist-Diagramm"""
        st.info("🚧 Nyquist-Diagramm wird in Schritt 4 implementiert...")
    
    def _create_root_locus(self, G_s, s):
        """Erstelle Wurzelortskurve"""
        st.info("🚧 Wurzelortskurve wird in Schritt 4 implementiert...")
    
    def _analyze_stability(self, G_s, s):
        """Analysiere Stabilität"""
        st.info("🚧 Stabilitätsanalyse wird in Schritt 5 implementiert...")
    
    def _calculate_stability_margins(self, G_s, s, omega):
        """Berechne Stabilitätsreserven"""
        st.info("🚧 Stabilitätsreserven werden in Schritt 5 implementiert...")
