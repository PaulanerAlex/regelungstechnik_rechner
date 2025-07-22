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
    from src.utils.display_utils import display_step_by_step, display_latex
except ImportError:
    def display_step_by_step(steps):
        """Fallback function for displaying steps"""
        for step_name, step_content in steps:
            st.markdown(f"**{step_name}:**")
            st.latex(step_content)
    
    def display_latex(content):
        """Fallback function for displaying LaTeX"""
        st.latex(content)

try:
    from src.utils.safe_sympify import safe_sympify
except ImportError:
    def safe_sympify(expr, symbols_dict=None):
        """Fallback safe_sympify function"""
        import sympy as sp
        if symbols_dict is None:
            symbols_dict = {}
        
        # Einfache automatische Multiplikation
        import re
        expr_str = str(expr)
        
        # Ersetze Muster wie "2s" mit "2*s"
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
        
        # Ersetze Muster wie "s(" mit "s*("
        expr_str = re.sub(r'([a-zA-Z])\(', r'\1*(', expr_str)
        
        # Ersetze Muster wie ")(" mit ")*(" für (s+1)(s+2) -> (s+1)*(s+2)
        expr_str = re.sub(r'\)\s*\(', r')*(', expr_str)
        
        # Ersetze Muster wie ")(s" mit ")*(s"
        expr_str = re.sub(r'\)([a-zA-Z])', r')*\1', expr_str)
        
        try:
            return sp.sympify(expr_str, locals=symbols_dict)
        except Exception as e:
            # Fallback: versuche es ohne Transformationen
            try:
                return sp.sympify(expr, locals=symbols_dict)
            except Exception:
                # Als letzter Ausweg: Fehlermeldung
                raise ValueError(f"Konnte '{expr}' nicht parsen. Versuchen Sie explizite Multiplikation (*) zu verwenden.")

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
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "🎯 Komplettanalyse",
                "📊 Pol-Nullstellen", 
                "📈 Frequenzgang",
                "🔄 Nyquist",
                "🌿 Wurzelortskurve",
                "⚖️ Stabilität",
                "📍 Ortskurve"
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
            
            with tab7:
                self.ortskurve_analysis()
    
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
                auto_simplify = st.checkbox("Automatisch vereinfachen", value=True, key='auto_simplify_main')
                show_steps = st.checkbox("Parsing-Schritte zeigen", value=False, key='show_steps_main')
        
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
                    # Konvertiere SymPy Objekt zu Python int
                    num_deg_val = int(num_deg) if num_deg is not None else 0
                    st.metric("Zählergrad", num_deg_val)
                
                with col_info2:
                    den_deg = sp.degree(sp.denom(G_s), st.session_state.tf_symbols['s'])
                    # Konvertiere SymPy Objekt zu Python int
                    den_deg_val = int(den_deg) if den_deg is not None else 0
                    st.metric("Nennergrad", den_deg_val)
                
                with col_info3:
                    system_type = "Eigentlich" if num_deg_val <= den_deg_val else "Uneigentlich"
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
            show_properties = st.checkbox("📋 Systemeigenschaften", value=True, key='show_properties_complete')
            show_poles_zeros = st.checkbox("📊 Pol-Nullstellen-Diagramm", value=True, key='show_poles_zeros_complete')
        
        with col2:
            show_frequency = st.checkbox("📈 Frequenzgang (Bode)", value=True, key='show_frequency_complete')
            show_nyquist = st.checkbox("🔄 Nyquist-Diagramm", value=True, key='show_nyquist_complete')
        
        with col3:
            show_stability = st.checkbox("⚖️ Stabilitätsanalyse", value=True, key='show_stability_complete')
            show_margins = st.checkbox("📏 Stabilitätsreserven", value=True, key='show_margins_complete')
        
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
            
            # Konvertiere SymPy Objekte zu Python ints
            num_degree_val = int(num_degree) if num_degree is not None else 0
            den_degree_val = int(den_degree) if den_degree is not None else 0
            
            st.markdown("**Systemparameter:**")
            st.write(f"• Zählergrad: {num_degree_val}")
            st.write(f"• Nennergrad: {den_degree_val}")
            st.write(f"• Systemordnung: {den_degree_val}")
            st.write(f"• Systemtyp: {'Eigentlich' if num_degree_val <= den_degree_val else 'Uneigentlich'}")
        
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
                num_degree_raw = sp.degree(num, s)
                den_degree_raw = sp.degree(den, s)
                num_degree = int(num_degree_raw) if num_degree_raw is not None else 0
                den_degree = int(den_degree_raw) if den_degree_raw is not None else 0
                
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
            if st.checkbox("🔬 Erweiterte Analyse anzeigen", key='extended_analysis_pz'):
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
        
        # Konfiguration der Frequenzgang-Analyse
        st.markdown("### ⚙️ Analyse-Einstellungen")
        
        col_settings1, col_settings2, col_settings3 = st.columns(3)
        
        with col_settings1:
            st.markdown("**Frequenzbereich:**")
            freq_min = st.number_input("ω min [rad/s]:", value=0.01, format="%.3f", min_value=0.001)
            freq_max = st.number_input("ω max [rad/s]:", value=100.0, format="%.1f", min_value=0.1)
        
        with col_settings2:
            st.markdown("**Plot-Optionen:**")
            show_bode = st.checkbox("📊 Bode-Diagramm", value=True, key='show_bode_freq')
            show_nyquist = st.checkbox("🔄 Nyquist-Plot", value=True, key='show_nyquist_freq')
            show_magnitude_phase = st.checkbox("📈 Getrennte Mag/Phase", value=False, key='show_magnitude_phase_freq')
        
        with col_settings3:
            st.markdown("**Zusatzanalyse:**")
            show_margins = st.checkbox("📏 Stabilitätsreserven", value=True, key='show_margins_freq')
            show_bandwidth = st.checkbox("⚡ Bandbreite", value=True, key='show_bandwidth_freq')
            auto_range = st.checkbox("🎯 Auto-Frequenzbereich", value=True, key='auto_range_freq')
        
        if st.button("� Frequenzgang analysieren", type="primary"):
            
            # Automatische Frequenzbereichsbestimmung
            if auto_range:
                freq_min, freq_max = self._determine_frequency_range(G_s, s)
                st.info(f"🎯 Automatischer Frequenzbereich: {freq_min:.3f} - {freq_max:.1f} rad/s")
            
            # Frequenzvektor erstellen
            omega_vals = np.logspace(np.log10(freq_min), np.log10(freq_max), 1000)
            
            # G(jω) berechnen
            j = sp.I
            G_jw = G_s.subs(s, j*omega)
            
            try:
                # Numerische Auswertung
                if G_jw.is_number:
                    # Konstante Übertragungsfunktion
                    G_vals = np.full_like(omega_vals, complex(G_jw), dtype=complex)
                else:
                    G_func = sp.lambdify(omega, G_jw, 'numpy')
                    G_vals = G_func(omega_vals)
                
                # Filter infinite/NaN values
                finite_mask = np.isfinite(G_vals)
                omega_clean = omega_vals[finite_mask]
                G_clean = G_vals[finite_mask]
                
                if len(G_clean) == 0:
                    st.error("❌ Keine gültigen Frequenzgang-Werte gefunden!")
                    return
                
                # Magnitude und Phase berechnen
                magnitude = np.abs(G_clean)
                phase_rad = np.angle(G_clean)
                phase_deg = phase_rad * 180 / np.pi
                magnitude_db = 20 * np.log10(magnitude)
                
                # Plots erstellen
                if show_bode:
                    st.markdown("---")
                    st.markdown("### 📊 Bode-Diagramm")
                    self._create_bode_plot_detailed(omega_clean, magnitude_db, phase_deg)
                
                if show_nyquist:
                    st.markdown("---")
                    st.markdown("### 🔄 Nyquist-Diagramm")
                    self._create_nyquist_plot_detailed(G_clean, omega_clean)
                
                if show_magnitude_phase:
                    st.markdown("---")
                    st.markdown("### 📈 Separierte Magnitude/Phase Darstellung")
                    self._create_separate_magnitude_phase_plots(omega_clean, magnitude, phase_deg)
                
                # Zusatzanalysen
                if show_margins:
                    st.markdown("---")
                    st.markdown("### 📏 Stabilitätsreserven")
                    self._analyze_stability_margins_detailed(omega_clean, magnitude_db, phase_deg)
                
                if show_bandwidth:
                    st.markdown("---")
                    st.markdown("### ⚡ Bandbreiten-Analyse")
                    self._analyze_bandwidth(omega_clean, magnitude_db, G_s, s)
                
            except Exception as e:
                st.error(f"❌ Fehler bei der Frequenzgang-Berechnung: {e}")
                import traceback
                with st.expander("Debug-Informationen"):
                    st.code(traceback.format_exc())
    
    def nyquist_analysis(self):
        """Detaillierte Nyquist-Analyse"""
        st.subheader("🔄 Nyquist-Analyse")
        
        if st.session_state.tf_parsed is None:
            st.warning("⚠️ Bitte definieren Sie zunächst eine Übertragungsfunktion.")
            return
        
        G_s = st.session_state.tf_parsed['G_s']
        s = st.session_state.tf_symbols['s']
        omega = st.session_state.tf_symbols['omega']
        
        # Konfiguration der Nyquist-Analyse
        st.markdown("### ⚙️ Nyquist-Einstellungen")
        
        col_nyq1, col_nyq2, col_nyq3 = st.columns(3)
        
        with col_nyq1:
            st.markdown("**Frequenzbereich:**")
            nyq_freq_min = st.number_input("ω min [rad/s]:", value=0.01, format="%.3f", min_value=0.001, key='nyq_freq_min')
            nyq_freq_max = st.number_input("ω max [rad/s]:", value=100.0, format="%.1f", min_value=0.1, key='nyq_freq_max')
            nyq_auto_range = st.checkbox("🎯 Auto-Frequenzbereich", value=True, key='nyq_auto_range')
        
        with col_nyq2:
            st.markdown("**Visualisierung:**")
            show_unit_circle = st.checkbox("⭕ Einheitskreis", value=True, key='show_unit_circle_nyq')
            show_critical_point = st.checkbox("❌ Kritischer Punkt (-1+0j)", value=True, key='show_critical_point_nyq')
            show_encirclements = st.checkbox("🔄 Umkreisungsanalyse", value=True, key='show_encirclements_nyq')
        
        with col_nyq3:
            st.markdown("**Erweiterte Optionen:**")
            include_negative_freq = st.checkbox("⚡ Negative Frequenzen", value=False, key='include_negative_freq_nyq')
            show_arrows = st.checkbox("➡️ Richtungspfeile", value=True, key='show_arrows_nyq')
            detailed_analysis = st.checkbox("🔬 Detaillierte Analyse", value=True, key='detailed_analysis_nyq')
        
        if st.button("� Nyquist-Analyse durchführen", type="primary", key='nyquist_analysis_btn'):
            
            # Automatische Frequenzbereichsbestimmung
            if nyq_auto_range:
                nyq_freq_min, nyq_freq_max = self._determine_frequency_range(G_s, s)
                st.info(f"🎯 Automatischer Frequenzbereich: {nyq_freq_min:.3f} - {nyq_freq_max:.1f} rad/s")
            
            try:
                # Erweiterte Nyquist-Berechnung
                self._create_advanced_nyquist_plot(
                    G_s, s, omega, nyq_freq_min, nyq_freq_max, 
                    include_negative_freq, show_unit_circle, show_critical_point, 
                    show_encirclements, show_arrows, detailed_analysis
                )
                
            except Exception as e:
                st.error(f"❌ Fehler bei der Nyquist-Analyse: {e}")
                import traceback
                with st.expander("Debug-Informationen"):
                    st.code(traceback.format_exc())
    
    def root_locus_analysis(self):
        """Detaillierte Wurzelortskurven-Analyse"""
        st.subheader("🌿 Wurzelortskurven-Analyse")
        
        if st.session_state.tf_parsed is None:
            st.warning("⚠️ Bitte definieren Sie zunächst eine Übertragungsfunktion.")
            return
        
        G_s = st.session_state.tf_parsed['G_s']
        s = st.session_state.tf_symbols['s']
        
        # Konfiguration der Wurzelortskurven-Analyse
        st.markdown("### ⚙️ Wurzelortskurven-Einstellungen")
        
        col_rl1, col_rl2, col_rl3 = st.columns(3)
        
        with col_rl1:
            st.markdown("**Parameter-Bereich:**")
            k_min = st.number_input("K min:", value=0.0, format="%.2f", key='k_min_rl')
            k_max = st.number_input("K max:", value=10.0, format="%.1f", min_value=0.1, key='k_max_rl')
            k_points = st.number_input("Anzahl K-Werte:", value=200, min_value=50, max_value=1000, key='k_points_rl')
        
        with col_rl2:
            st.markdown("**Visualisierung:**")
            show_poles_zeros_rl = st.checkbox("📍 Pole/Nullstellen", value=True, key='show_poles_zeros_rl')
            show_asymptotes = st.checkbox("📐 Asymptoten", value=True, key='show_asymptotes_rl')
            show_breakpoints = st.checkbox("🔗 Verzweigungspunkte", value=True, key='show_breakpoints_rl')
        
        with col_rl3:
            st.markdown("**Erweiterte Optionen:**")
            show_k_values = st.checkbox("🏷️ K-Werte anzeigen", value=False, key='show_k_values_rl')
            interactive_k = st.checkbox("🎛️ Interaktive K-Variation", value=True, key='interactive_k_rl')
            stability_regions = st.checkbox("�️ Stabilitätsregionen", value=True, key='stability_regions_rl')
        
        if st.button("🌿 Wurzelortskurven-Analyse durchführen", type="primary", key='root_locus_analysis_btn'):
            
            try:
                # Erweiterte Wurzelortskurven-Berechnung
                self._create_advanced_root_locus_plot(
                    G_s, s, k_min, k_max, k_points,
                    show_poles_zeros_rl, show_asymptotes, show_breakpoints,
                    show_k_values, interactive_k, stability_regions
                )
                
            except Exception as e:
                st.error(f"❌ Fehler bei der Wurzelortskurven-Analyse: {e}")
                import traceback
                with st.expander("Debug-Informationen"):
                    st.code(traceback.format_exc())
    
    def stability_analysis(self):
        """Detaillierte Stabilitätsanalyse"""
        st.subheader("⚖️ Stabilitätsanalyse")
        
        if st.session_state.tf_parsed is None:
            st.warning("⚠️ Bitte definieren Sie zunächst eine Übertragungsfunktion.")
            return
        
        G_s = st.session_state.tf_parsed['G_s']
        s = st.session_state.tf_symbols['s']
        omega = st.session_state.tf_symbols['omega']
        
        # Konfiguration der Stabilitätsanalyse
        st.markdown("### ⚙️ Stabilitäts-Einstellungen")
        
        col_stab1, col_stab2, col_stab3 = st.columns(3)
        
        with col_stab1:
            st.markdown("**Analysemethoden:**")
            show_pole_analysis = st.checkbox("📍 Pol-Analyse", value=True, key='show_pole_analysis_stab')
            show_routh_hurwitz = st.checkbox("📋 Routh-Hurwitz", value=True, key='show_routh_hurwitz_stab')
            show_nyquist_criterion = st.checkbox("🔄 Nyquist-Kriterium", value=True, key='show_nyquist_criterion_stab')
        
        with col_stab2:
            st.markdown("**Stabilitätsreserven:**")
            show_gain_margin = st.checkbox("📊 Amplitudenreserve", value=True, key='show_gain_margin_stab')
            show_phase_margin = st.checkbox("📐 Phasenreserve", value=True, key='show_phase_margin_stab')
            show_sensitivity = st.checkbox("🎯 Sensitivitätsanalyse", value=True, key='show_sensitivity_stab')
        
        with col_stab3:
            st.markdown("**Robustheitsanalyse:**")
            show_robustness = st.checkbox("🛡️ Robustheit", value=True, key='show_robustness_stab')
            show_performance = st.checkbox("⚡ Performance-Metriken", value=True, key='show_performance_stab')
            show_recommendations = st.checkbox("� Empfehlungen", value=True, key='show_recommendations_stab')
        
        if st.button("⚖️ Vollständige Stabilitätsanalyse", type="primary", key='stability_analysis_btn'):
            
            try:
                # Umfassende Stabilitätsanalyse
                self._perform_comprehensive_stability_analysis(
                    G_s, s, omega,
                    show_pole_analysis, show_routh_hurwitz, show_nyquist_criterion,
                    show_gain_margin, show_phase_margin, show_sensitivity,
                    show_robustness, show_performance, show_recommendations
                )
                
            except Exception as e:
                st.error(f"❌ Fehler bei der Stabilitätsanalyse: {e}")
                import traceback
                with st.expander("Debug-Informationen"):
                    st.code(traceback.format_exc())
    
    def ortskurve_analysis(self):
        """Ortskurven-Analyse"""
        st.subheader("📍 Ortskurven-Analyse")
        
        if st.session_state.tf_parsed is None:
            st.warning("⚠️ Bitte definieren Sie zunächst eine Übertragungsfunktion.")
            return
        
        G_s = st.session_state.tf_parsed['G_s']
        s = st.session_state.tf_symbols['s']
        omega = st.session_state.tf_symbols['omega']
        
        st.markdown("""
        **Ortskurve**: Darstellung von G(jω) für verschiedene Frequenzen ω im komplexen Zahlenbereich.
        Im Gegensatz zur Wurzelortskurve wird hier die Frequenz variiert, nicht der Verstärkungsfaktor.
        """)
        
        # Konfiguration der Ortskurven-Analyse
        st.markdown("### ⚙️ Ortskurven-Einstellungen")
        
        col_ort1, col_ort2, col_ort3 = st.columns(3)
        
        with col_ort1:
            st.markdown("**Frequenzbereich:**")
            ort_freq_min = st.number_input("ω min [rad/s]:", value=0.01, format="%.3f", min_value=0.001, key='ort_freq_min')
            ort_freq_max = st.number_input("ω max [rad/s]:", value=100.0, format="%.1f", min_value=0.1, key='ort_freq_max')
            ort_auto_range = st.checkbox("🎯 Auto-Frequenzbereich", value=True, key='ort_auto_range')
        
        with col_ort2:
            st.markdown("**Visualisierung:**")
            show_freq_markers = st.checkbox("🏷️ Frequenz-Markierungen", value=True, key='show_freq_markers_ort')
            show_unit_circle_ort = st.checkbox("⭕ Einheitskreis", value=True, key='show_unit_circle_ort')
            show_grid_ort = st.checkbox("📊 Raster", value=True, key='show_grid_ort')
        
        with col_ort3:
            st.markdown("**Optionen:**")
            ortskurve_points = st.number_input("Anzahl Punkte:", value=500, min_value=100, max_value=2000, key='ortskurve_points')
            color_by_frequency = st.checkbox("🌈 Farbcodierung nach ω", value=True, key='color_by_frequency_ort')
            show_direction = st.checkbox("➡️ Richtungspfeile", value=False, key='show_direction_ort')
        
        if st.button("📍 Ortskurve erstellen", type="primary", key='ortskurve_analysis_btn'):
            
            # Automatische Frequenzbereichsbestimmung
            if ort_auto_range:
                ort_freq_min, ort_freq_max = self._determine_frequency_range(G_s, s)
                st.info(f"🎯 Automatischer Frequenzbereich: {ort_freq_min:.3f} - {ort_freq_max:.1f} rad/s")
            
            try:
                # Ortskurven-Berechnung und Darstellung
                self._create_ortskurve_plot(
                    G_s, s, omega, ort_freq_min, ort_freq_max, ortskurve_points,
                    show_freq_markers, show_unit_circle_ort, show_grid_ort,
                    color_by_frequency, show_direction
                )
                
            except Exception as e:
                st.error(f"❌ Fehler bei der Ortskurven-Analyse: {e}")
                import traceback
                with st.expander("Debug-Informationen"):
                    st.code(traceback.format_exc())
    
    def _determine_frequency_range(self, G_s, s):
        """Bestimme automatisch einen sinnvollen Frequenzbereich"""
        try:
            # Pole extrahieren
            poles = sp.solve(sp.denom(G_s), s)
            
            if not poles:
                return 0.01, 100.0  # Standard-Bereich
            
            # Charakteristische Frequenzen bestimmen
            char_freqs = []
            for pole in poles:
                pole_val = complex(pole.evalf())
                if pole_val.real < 0:  # Nur stabile Pole
                    char_freqs.append(abs(pole_val))
            
            # Nullstellen hinzufügen
            zeros = sp.solve(sp.numer(G_s), s)
            for zero in zeros:
                zero_val = complex(zero.evalf())
                char_freqs.append(abs(zero_val))
            
            if char_freqs:
                min_freq = min(char_freqs) / 100  # Eine Dekade unter niedrigster Freq
                max_freq = max(char_freqs) * 100  # Eine Dekade über höchster Freq
                
                # Sinnvolle Grenzen
                min_freq = max(min_freq, 0.001)
                max_freq = min(max_freq, 10000)
                
                return min_freq, max_freq
            else:
                return 0.01, 100.0
                
        except Exception:
                return 0.01, 100.0  # Fallback
    
    def _create_ortskurve_plot(self, G_s, s, omega, freq_min, freq_max, num_points,
                              show_freq_markers, show_unit_circle, show_grid,
                              color_by_frequency, show_direction):
        """Erstelle Ortskurven-Plot"""
        
        # Frequenzvektor erstellen
        omega_vals = np.logspace(np.log10(freq_min), np.log10(freq_max), num_points)
        
        # G(jω) berechnen
        j = sp.I
        G_jw = G_s.subs(s, j*omega)
        
        try:
            if G_jw.is_number:
                G_vals = np.full_like(omega_vals, complex(G_jw), dtype=complex)
            else:
                G_func = sp.lambdify(omega, G_jw, 'numpy')
                G_vals = G_func(omega_vals)
            
            # Filter infinite/NaN values
            finite_mask = np.isfinite(G_vals)
            omega_clean = omega_vals[finite_mask]
            G_clean = G_vals[finite_mask]
            
            if len(G_clean) == 0:
                st.error("❌ Keine gültigen Werte für Ortskurve!")
                return
            
            real_vals = np.real(G_clean)
            imag_vals = np.imag(G_clean)
            
            # Plot erstellen
            fig = go.Figure()
            
            # Hauptortskurve
            if color_by_frequency:
                # Farbcodierung nach Frequenz
                fig.add_trace(go.Scatter(
                    x=real_vals, y=imag_vals,
                    mode='lines+markers',
                    line=dict(width=3),
                    marker=dict(
                        size=4,
                        color=np.log10(omega_clean),
                        colorscale='Viridis',
                        colorbar=dict(
                            title=dict(text="log₁₀(ω)")
                        ),
                        showscale=True
                    ),
                    name='G(jω)',
                    hovertemplate='ω: %{text:.3f} rad/s<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>',
                    text=omega_clean
                ))
            else:
                # Einfarbige Darstellung
                fig.add_trace(go.Scatter(
                    x=real_vals, y=imag_vals,
                    mode='lines+markers',
                    line=dict(color='blue', width=3),
                    marker=dict(size=4, color='blue'),
                    name='G(jω)',
                    hovertemplate='ω: %{text:.3f} rad/s<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>',
                    text=omega_clean
                ))
            
            # Frequenz-Markierungen
            if show_freq_markers:
                # Markiere charakteristische Frequenzen
                marker_frequencies = [freq_min, freq_max]
                
                # Füge Dekaden-Frequenzen hinzu
                decade_freqs = []
                current_decade = 10 ** np.floor(np.log10(freq_min))
                while current_decade <= freq_max:
                    if current_decade >= freq_min:
                        decade_freqs.append(current_decade)
                    current_decade *= 10
                
                marker_frequencies.extend(decade_freqs)
                marker_frequencies = sorted(list(set(marker_frequencies)))
                
                for freq in marker_frequencies:
                    if freq_min <= freq <= freq_max:
                        # Interpoliere G(jω) bei dieser Frequenz
                        idx = np.argmin(np.abs(omega_clean - freq))
                        marker_real = real_vals[idx]
                        marker_imag = imag_vals[idx]
                        
                        fig.add_trace(go.Scatter(
                            x=[marker_real], y=[marker_imag],
                            mode='markers+text',
                            marker=dict(size=8, color='red', symbol='circle'),
                            text=[f'ω={freq:.2f}'],
                            textposition="top center",
                            name=f'ω={freq:.2f}',
                            showlegend=False,
                            hovertemplate=f'ω: {freq:.3f} rad/s<br>Real: {marker_real:.3f}<br>Imag: {marker_imag:.3f}<extra></extra>'
                        ))
            
            # Richtungspfeile
            if show_direction and len(real_vals) > 10:
                # Füge Pfeile hinzu um Richtung zu zeigen
                arrow_indices = np.linspace(0, len(real_vals)-2, min(10, len(real_vals)//10), dtype=int)
                
                for i in arrow_indices:
                    if i < len(real_vals) - 1:
                        dx = real_vals[i+1] - real_vals[i]
                        dy = imag_vals[i+1] - imag_vals[i]
                        
                        fig.add_annotation(
                            x=real_vals[i],
                            y=imag_vals[i],
                            ax=real_vals[i] + dx * 0.1,
                            ay=imag_vals[i] + dy * 0.1,
                            xref="x", yref="y",
                            axref="x", ayref="y",
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="darkblue",
                            showarrow=True
                        )
            
            # Einheitskreis
            if show_unit_circle:
                theta = np.linspace(0, 2*np.pi, 100)
                unit_circle_x = np.cos(theta)
                unit_circle_y = np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=unit_circle_x, y=unit_circle_y,
                    mode='lines',
                    line=dict(color='lightgray', width=1, dash='dot'),
                    name='Einheitskreis',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Koordinatenachsen
            fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.3)
            fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.3)
            
            # Startpunkt (niedrigste Frequenz) markieren
            fig.add_trace(go.Scatter(
                x=[real_vals[0]], y=[imag_vals[0]],
                mode='markers',
                marker=dict(size=12, color='green', symbol='diamond'),
                name=f'Start (ω={omega_clean[0]:.3f})',
                hovertemplate=f'Start<br>ω: {omega_clean[0]:.3f} rad/s<br>Real: {real_vals[0]:.3f}<br>Imag: {imag_vals[0]:.3f}<extra></extra>'
            ))
            
            # Endpunkt (höchste Frequenz) markieren
            fig.add_trace(go.Scatter(
                x=[real_vals[-1]], y=[imag_vals[-1]],
                mode='markers',
                marker=dict(size=12, color='red', symbol='square'),
                name=f'Ende (ω={omega_clean[-1]:.3f})',
                hovertemplate=f'Ende<br>ω: {omega_clean[-1]:.3f} rad/s<br>Real: {real_vals[-1]:.3f}<br>Imag: {imag_vals[-1]:.3f}<extra></extra>'
            ))
            
            # Layout konfigurieren
            x_range = [min(real_vals), max(real_vals)]
            y_range = [min(imag_vals), max(imag_vals)]
            
            # Padding hinzufügen
            x_padding = (x_range[1] - x_range[0]) * 0.1
            y_padding = (y_range[1] - y_range[0]) * 0.1
            
            fig.update_layout(
                title="Ortskurve G(jω)",
                xaxis_title="Realteil",
                yaxis_title="Imaginärteil",
                showlegend=True,
                width=800, height=600,
                xaxis=dict(
                    range=[x_range[0]-x_padding, x_range[1]+x_padding],
                    showgrid=show_grid,
                    scaleanchor="y",
                    scaleratio=1
                ),
                yaxis=dict(
                    range=[y_range[0]-y_padding, y_range[1]+y_padding],
                    showgrid=show_grid
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Zusätzliche Ortskurven-Analyse
            self._ortskurve_additional_analysis(omega_clean, real_vals, imag_vals, G_s, s)
            
        except Exception as e:
            st.error(f"Fehler beim Erstellen der Ortskurve: {e}")
            import traceback
            with st.expander("Debug-Informationen"):
                st.code(traceback.format_exc())
    
    def _ortskurve_additional_analysis(self, omega_vals, real_vals, imag_vals, G_s, s):
        """Zusätzliche Ortskurven-Analyse"""
        
        st.markdown("---")
        st.markdown("### 📊 Ortskurven-Charakteristik")
        
        col_ort1, col_ort2, col_ort3 = st.columns(3)
        
        with col_ort1:
            st.markdown("**Extremwerte:**")
            
            # Maximale und minimale Real-/Imaginärteile
            max_real_idx = np.argmax(real_vals)
            min_real_idx = np.argmin(real_vals)
            max_imag_idx = np.argmax(imag_vals)
            min_imag_idx = np.argmin(imag_vals)
            
            st.write(f"**Max Real:** {real_vals[max_real_idx]:.3f}")
            st.write(f"  bei ω = {omega_vals[max_real_idx]:.3f} rad/s")
            st.write(f"**Min Real:** {real_vals[min_real_idx]:.3f}")
            st.write(f"  bei ω = {omega_vals[min_real_idx]:.3f} rad/s")
            
        with col_ort2:
            st.write(f"**Max Imag:** {imag_vals[max_imag_idx]:.3f}")
            st.write(f"  bei ω = {omega_vals[max_imag_idx]:.3f} rad/s")
            st.write(f"**Min Imag:** {imag_vals[min_imag_idx]:.3f}")
            st.write(f"  bei ω = {omega_vals[min_imag_idx]:.3f} rad/s")
            
            # Maximaler Betrag
            magnitude = np.sqrt(real_vals**2 + imag_vals**2)
            max_mag_idx = np.argmax(magnitude)
            st.write(f"**Max Betrag:** {magnitude[max_mag_idx]:.3f}")
            st.write(f"  bei ω = {omega_vals[max_mag_idx]:.3f} rad/s")
        
        with col_ort3:
            st.markdown("**Verhalten:**")
            
            # Verhalten bei niedrigen/hohen Frequenzen
            st.write("**ω → 0:**")
            st.write(f"  G(j0) ≈ {real_vals[0]:.3f} + {imag_vals[0]:.3f}j")
            
            st.write("**ω → ∞:**")
            st.write(f"  G(j∞) ≈ {real_vals[-1]:.3f} + {imag_vals[-1]:.3f}j")
            
            # Kreuzungen mit Achsen
            real_axis_crossings = np.where(np.diff(np.signbit(imag_vals)))[0]
            imag_axis_crossings = np.where(np.diff(np.signbit(real_vals)))[0]
            
            st.write(f"**Achsenkreuzungen:**")
            st.write(f"  Reelle Achse: {len(real_axis_crossings)}")
            st.write(f"  Imaginäre Achse: {len(imag_axis_crossings)}")
            
            # Verhalten bei niedrigen/hohen Frequenzen
            st.write("**ω → 0:**")
            st.write(f"  G(j0) ≈ {real_vals[0]:.3f} + {imag_vals[0]:.3f}j")
            
            st.write("**ω → ∞:**")
            st.write(f"  G(j∞) ≈ {real_vals[-1]:.3f} + {imag_vals[-1]:.3f}j")
            
            # Kreuzungen mit Achsen
            real_axis_crossings = np.where(np.diff(np.signbit(imag_vals)))[0]
            imag_axis_crossings = np.where(np.diff(np.signbit(real_vals)))[0]
            
            st.write(f"**Achsenkreuzungen:**")
            st.write(f"  Reelle Achse: {len(real_axis_crossings)}")
            st.write(f"  Imaginäre Achse: {len(imag_axis_crossings)}")
        
        # Spezielle Punkte
        st.markdown("#### 🎯 Charakteristische Punkte")
        
        # Prüfe ob Ortskurve durch kritischen Punkt geht
        distances_to_critical = np.sqrt((real_vals + 1)**2 + imag_vals**2)
        min_distance_to_critical = np.min(distances_to_critical)
        closest_idx = np.argmin(distances_to_critical)
        
        col_char1, col_char2 = st.columns(2)
        
        with col_char1:
            st.write(f"**Nächster Punkt zu (-1+0j):**")
            st.write(f"  Abstand: {min_distance_to_critical:.4f}")
            st.write(f"  bei ω = {omega_vals[closest_idx]:.3f} rad/s")
            st.write(f"  Punkt: {real_vals[closest_idx]:.3f} + {imag_vals[closest_idx]:.3f}j")
        
        with col_char2:
            # Bewertung der Nähe zum kritischen Punkt
            if min_distance_to_critical < 0.1:
                st.error("❌ Sehr nah am kritischen Punkt!")
            elif min_distance_to_critical < 0.5:
                st.warning("⚠️ Nah am kritischen Punkt")
            elif min_distance_to_critical < 1.0:
                st.info("ℹ️ Mäßige Entfernung zum kritischen Punkt")
            else:
                st.success("✅ Sicherer Abstand zum kritischen Punkt")
    
    def _create_bode_plot_detailed(self, omega_vals, magnitude_db, phase_deg):
        """Erstelle detailliertes Bode-Diagramm"""
        
        # Erstelle Subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Amplitudengang |G(jω)| [dB]', 'Phasengang ∠G(jω) [°]'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Amplitudengang
        fig.add_trace(
            go.Scatter(
                x=omega_vals, y=magnitude_db,
                mode='lines',
                name='|G(jω)|',
                line=dict(color='blue', width=2),
                hovertemplate='ω: %{x:.3f} rad/s<br>|G|: %{y:.1f} dB<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Phasengang
        fig.add_trace(
            go.Scatter(
                x=omega_vals, y=phase_deg,
                mode='lines',
                name='∠G(jω)',
                line=dict(color='red', width=2),
                hovertemplate='ω: %{x:.3f} rad/s<br>∠G: %{y:.1f}°<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Referenzlinien
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="0 dB", row=1, col=1)
        fig.add_hline(y=-3, line_dash="dot", line_color="orange", 
                     annotation_text="-3 dB", row=1, col=1)
        fig.add_hline(y=-180, line_dash="dash", line_color="gray",
                     annotation_text="-180°", row=2, col=1)
        fig.add_hline(y=-90, line_dash="dot", line_color="orange",
                     annotation_text="-90°", row=2, col=1)
        
        # Layout konfigurieren
        fig.update_xaxes(type="log", title_text="Frequenz ω [rad/s]", row=2, col=1)
        fig.update_xaxes(type="log", row=1, col=1)
        fig.update_yaxes(title_text="Magnitude [dB]", row=1, col=1)
        fig.update_yaxes(title_text="Phase [°]", row=2, col=1)
        
        fig.update_layout(
            title="Bode-Diagramm",
            height=600,
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Wichtige Frequenzen markieren
        self._mark_important_frequencies(omega_vals, magnitude_db, phase_deg)
    
    def _create_nyquist_plot_detailed(self, G_vals, omega_vals):
        """Erstelle detailliertes Nyquist-Diagramm"""
        
        real_vals = np.real(G_vals)
        imag_vals = np.imag(G_vals)
        
        fig = go.Figure()
        
        # Hauptkurve
        fig.add_trace(go.Scatter(
            x=real_vals, y=imag_vals,
            mode='lines+markers',
            name='G(jω)',
            line=dict(color='blue', width=2),
            marker=dict(size=3),
            hovertemplate='ω: %{text}<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>',
            text=[f'{w:.2f}' for w in omega_vals]
        ))
        
        # Kritischer Punkt -1+0j
        fig.add_trace(go.Scatter(
            x=[-1], y=[0],
            mode='markers',
            marker=dict(symbol='x', size=15, color='red', line=dict(width=3)),
            name='Kritischer Punkt (-1+0j)',
            hovertemplate='Kritischer Punkt<br>Real: -1<br>Imag: 0<extra></extra>'
        ))
        
        # Einheitskreis zur Orientierung
        theta = np.linspace(0, 2*np.pi, 100)
        unit_circle_x = np.cos(theta)
        unit_circle_y = np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=unit_circle_x, y=unit_circle_y,
            mode='lines',
            line=dict(color='lightgray', width=1, dash='dot'),
            name='Einheitskreis',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Koordinatenachsen
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.3)
        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.3)
        
        # Auto-Skalierung mit gleichmäßigen Achsen
        x_range = [min(real_vals), max(real_vals)]
        y_range = [min(imag_vals), max(imag_vals)]
        
        # Erweitere Bereich um kritischen Punkt falls nötig
        x_range[0] = min(x_range[0], -1.5)
        x_range[1] = max(x_range[1], 0.5)
        
        # Padding hinzufügen
        x_padding = (x_range[1] - x_range[0]) * 0.1
        y_padding = (y_range[1] - y_range[0]) * 0.1
        
        fig.update_layout(
            title="Nyquist-Diagramm",
            xaxis_title="Realteil",
            yaxis_title="Imaginärteil",
            showlegend=True,
            width=700, height=600,
            xaxis=dict(
                range=[x_range[0]-x_padding, x_range[1]+x_padding],
                showgrid=True,
                scaleanchor="y", 
                scaleratio=1
            ),
            yaxis=dict(
                range=[y_range[0]-y_padding, y_range[1]+y_padding],
                showgrid=True
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Nyquist-Stabilitätsanalyse
        self._nyquist_stability_analysis(real_vals, imag_vals, omega_vals)
    
    def _mark_important_frequencies(self, omega_vals, magnitude_db, phase_deg):
        """Markiere wichtige Frequenzen im Bode-Diagramm"""
        st.markdown("#### 🎯 Charakteristische Frequenzen")
        
        col_freq1, col_freq2, col_freq3 = st.columns(3)
        
        with col_freq1:
            # Eckfrequenz (0 dB Durchgang)
            zero_crossings = np.where(np.diff(np.signbit(magnitude_db)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                if idx < len(omega_vals) - 1:
                    crossover_freq = np.interp(0, [magnitude_db[idx], magnitude_db[idx+1]], 
                                              [omega_vals[idx], omega_vals[idx+1]])
                    st.metric("🎯 Eckfrequenz", f"{crossover_freq:.3f} rad/s")
                else:
                    st.metric("🎯 Eckfrequenz", "Nicht gefunden")
            else:
                st.metric("🎯 Eckfrequenz", "Nicht gefunden")
        
        with col_freq2:
            # -3dB Frequenz (Bandbreite)
            db_3_crossings = np.where(np.diff(np.signbit(magnitude_db + 3)))[0]
            if len(db_3_crossings) > 0:
                idx = db_3_crossings[0] 
                if idx < len(omega_vals) - 1:
                    bandwidth_freq = np.interp(-3, [magnitude_db[idx], magnitude_db[idx+1]], 
                                              [omega_vals[idx], omega_vals[idx+1]])
                    st.metric("📡 -3dB Frequenz", f"{bandwidth_freq:.3f} rad/s")
                else:
                    st.metric("📡 -3dB Frequenz", "Nicht gefunden")
            else:
                st.metric("📡 -3dB Frequenz", "Nicht gefunden")
        
        with col_freq3:
            # Resonanzfrequenz (Maximum in Magnitude)
            max_idx = np.argmax(magnitude_db)
            resonance_freq = omega_vals[max_idx]
            resonance_mag = magnitude_db[max_idx]
            st.metric("⚡ Resonanzfrequenz", f"{resonance_freq:.3f} rad/s")
            st.write(f"   Max: {resonance_mag:.1f} dB")
    
    def _create_separate_magnitude_phase_plots(self, omega_vals, magnitude, phase_deg):
        """Erstelle separate Magnitude und Phase Plots"""
        
        col_mag, col_phase = st.columns(2)
        
        with col_mag:
            st.markdown("**Amplitudengang (linear)**")
            fig_mag = go.Figure()
            fig_mag.add_trace(go.Scatter(
                x=omega_vals, y=magnitude,
                mode='lines',
                line=dict(color='blue', width=2),
                hovertemplate='ω: %{x:.3f}<br>|G|: %{y:.3f}<extra></extra>'
            ))
            
            fig_mag.update_layout(
                xaxis_type="log",
                xaxis_title="ω [rad/s]",
                yaxis_title="|G(jω)|",
                height=400
            )
            st.plotly_chart(fig_mag, use_container_width=True)
        
        with col_phase:
            st.markdown("**Phasengang**")
            fig_phase = go.Figure()
            fig_phase.add_trace(go.Scatter(
                x=omega_vals, y=phase_deg,
                mode='lines',
                line=dict(color='red', width=2),
                hovertemplate='ω: %{x:.3f}<br>∠G: %{y:.1f}°<extra></extra>'
            ))
            
            fig_phase.add_hline(y=-180, line_dash="dash", line_color="gray")
            fig_phase.add_hline(y=-90, line_dash="dot", line_color="orange")
            
            fig_phase.update_layout(
                xaxis_type="log",
                xaxis_title="ω [rad/s]",
                yaxis_title="∠G(jω) [°]",
                height=400
            )
            st.plotly_chart(fig_phase, use_container_width=True)
    
    def _analyze_stability_margins_detailed(self, omega_vals, magnitude_db, phase_deg):
        """Detaillierte Stabilitätsreserven-Analyse"""
        
        col_margin1, col_margin2 = st.columns(2)
        
        with col_margin1:
            st.markdown("#### 📊 Amplitudenreserve")
            
            # Finde -180° Durchgang
            phase_shifted = phase_deg + 180
            phase_crossings = np.where(np.diff(np.signbit(phase_shifted)))[0]
            
            if len(phase_crossings) > 0:
                idx = phase_crossings[0]
                if idx < len(omega_vals) - 1:
                    phase_crossover_freq = np.interp(0, [phase_shifted[idx], phase_shifted[idx+1]], 
                                                   [omega_vals[idx], omega_vals[idx+1]])
                    # Magnitude bei dieser Frequenz
                    freq_idx = np.argmin(np.abs(omega_vals - phase_crossover_freq))
                    gain_margin = -magnitude_db[freq_idx]
                    
                    st.metric("Amplitudenreserve", f"{gain_margin:.1f} dB")
                    st.write(f"Bei ω = {phase_crossover_freq:.3f} rad/s")
                    
                    if gain_margin > 6:
                        st.success("✅ Gut (> 6 dB)")
                    elif gain_margin > 3:
                        st.warning("⚠️ Akzeptabel (3-6 dB)")
                    else:
                        st.error("❌ Kritisch (< 3 dB)")
                else:
                    st.write("Nicht berechenbar")
            else:
                st.write("Kein -180° Durchgang gefunden")
        
        with col_margin2:
            st.markdown("#### � Phasenreserve")
            
            # Finde 0 dB Durchgang
            zero_crossings = np.where(np.diff(np.signbit(magnitude_db)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                if idx < len(omega_vals) - 1:
                    gain_crossover_freq = np.interp(0, [magnitude_db[idx], magnitude_db[idx+1]], 
                                                  [omega_vals[idx], omega_vals[idx+1]])
                    # Phase bei dieser Frequenz
                    freq_idx = np.argmin(np.abs(omega_vals - gain_crossover_freq))
                    phase_at_crossover = phase_deg[freq_idx]
                    phase_margin = 180 + phase_at_crossover
                    
                    st.metric("Phasenreserve", f"{phase_margin:.1f}°")
                    st.write(f"Bei ω = {gain_crossover_freq:.3f} rad/s")
                    
                    if phase_margin > 45:
                        st.success("✅ Gut (> 45°)")
                    elif phase_margin > 30:
                        st.warning("⚠️ Akzeptabel (30-45°)")
                    else:
                        st.error("❌ Kritisch (< 30°)")
                else:
                    st.write("Nicht berechenbar")
            else:
                st.write("Kein 0 dB Durchgang gefunden")
    
    def _analyze_bandwidth(self, omega_vals, magnitude_db, G_s, s):
        """Bandbreiten-Analyse"""
        
        col_bw1, col_bw2 = st.columns(2)
        
        with col_bw1:
            st.markdown("#### 📡 Bandbreite")
            
            # DC-Verstärkung
            try:
                dc_gain_db = magnitude_db[0]  # Erste Frequenz (niedrigste)
                cutoff_level = dc_gain_db - 3  # -3dB Punkt
                
                # Finde -3dB Punkt
                below_cutoff = magnitude_db < cutoff_level
                if np.any(below_cutoff):
                    cutoff_idx = np.where(below_cutoff)[0][0]
                    if cutoff_idx > 0:
                        bandwidth = np.interp(cutoff_level, 
                                            [magnitude_db[cutoff_idx-1], magnitude_db[cutoff_idx]],
                                            [omega_vals[cutoff_idx-1], omega_vals[cutoff_idx]])
                        st.metric("Bandbreite (-3dB)", f"{bandwidth:.3f} rad/s")
                        st.write(f"= {bandwidth/(2*np.pi):.3f} Hz")
                    else:
                        st.write("Bandbreite > Analysierte Frequenz")
                else:
                    st.write("Bandbreite > Analysierte Frequenz")
                    
            except Exception as e:
                st.write(f"Berechnung nicht möglich: {e}")
        
        with col_bw2:
            st.markdown("#### ⚡ Systemperformance")
            
            # Einschwingzeit schätzen
            try:
                poles = sp.solve(sp.denom(G_s), s)
                if poles:
                    # Dominanter Pol (nächster zur imaginären Achse)
                    pole_reals = [abs(complex(pole.evalf()).real) for pole in poles 
                                 if complex(pole.evalf()).real < 0]
                    if pole_reals:
                        dominant_real = min(pole_reals)
                        settling_time = 4 / dominant_real  # 2% Kriterium
                        st.metric("Einschwingzeit (2%)", f"{settling_time:.3f} s")
                        
                        rise_time = 2.2 / dominant_real  # Näherung
                        st.metric("Anstiegszeit (10-90%)", f"{rise_time:.3f} s")
                    else:
                        st.write("Keine stabilen Pole")
                else:
                    st.write("Keine Pole gefunden")
            except Exception as e:
                st.write(f"Berechnung nicht möglich: {e}")
    
    def _nyquist_stability_analysis(self, real_vals, imag_vals, omega_vals):
        """Nyquist-Stabilitätsanalyse"""
        st.markdown("#### 🔍 Nyquist-Stabilitätskriterium")
        
        # Anzahl Umkreisungen des kritischen Punktes (-1+0j)
        try:
            # Vereinfachte Umkreisungsanalyse
            critical_point = np.array([-1, 0])
            
            # Berechne Winkeländerung
            vectors_to_critical = np.column_stack([real_vals + 1, imag_vals])
            angles = np.arctan2(vectors_to_critical[:, 1], vectors_to_critical[:, 0])
            
            # Unwrap angles und berechne Gesamtwinkeländerung
            angles_unwrapped = np.unwrap(angles)
            total_angle_change = angles_unwrapped[-1] - angles_unwrapped[0]
            encirclements = total_angle_change / (2 * np.pi)
            
            col_nyq1, col_nyq2 = st.columns(2)
            
            with col_nyq1:
                st.metric("Umkreisungen", f"{encirclements:.1f}")
                
                # Minimaler Abstand zum kritischen Punkt
                distances = np.sqrt((real_vals + 1)**2 + imag_vals**2)
                min_distance = np.min(distances)
                min_idx = np.argmin(distances)
                
                st.metric("Min. Abstand zu (-1+0j)", f"{min_distance:.3f}")
                st.write(f"Bei ω = {omega_vals[min_idx]:.3f} rad/s")
            
            with col_nyq2:
                if abs(encirclements) < 0.1:
                    st.success("✅ Keine Umkreisungen - System stabil")
                else:
                    st.warning(f"⚠️ {abs(encirclements):.1f} Umkreisungen detektiert")
                
                if min_distance < 0.5:
                    st.warning("⚠️ Nahe am kritischen Punkt!")
                elif min_distance < 1.0:
                    st.info("ℹ️ Moderate Nähe zum kritischen Punkt")
                else:
                    st.success("✅ Sicherer Abstand zum kritischen Punkt")
                    
        except Exception as e:
            st.write(f"Umkreisungsanalyse nicht möglich: {e}")
    # Aktualisierte Methoden für die Komplettanalyse
    def _create_bode_plot(self, G_s, s, omega):
        """Erstelle Bode-Diagramm für Komplettanalyse"""
        try:
            # Automatischer Frequenzbereich
            freq_min, freq_max = self._determine_frequency_range(G_s, s)
            omega_vals = np.logspace(np.log10(freq_min), np.log10(freq_max), 500)
            
            # G(jω) berechnen
            j = sp.I
            G_jw = G_s.subs(s, j*omega)
            
            if G_jw.is_number:
                G_vals = np.full_like(omega_vals, complex(G_jw), dtype=complex)
            else:
                G_func = sp.lambdify(omega, G_jw, 'numpy')
                G_vals = G_func(omega_vals)
            
            # Filter infinite values
            finite_mask = np.isfinite(G_vals)
            omega_clean = omega_vals[finite_mask]
            G_clean = G_vals[finite_mask]
            
            if len(G_clean) > 0:
                magnitude_db = 20 * np.log10(np.abs(G_clean))
                phase_deg = np.angle(G_clean) * 180 / np.pi
                self._create_bode_plot_detailed(omega_clean, magnitude_db, phase_deg)
            else:
                st.error("Keine gültigen Frequenzgang-Werte")
                
        except Exception as e:
            st.error(f"Bode-Diagramm Fehler: {e}")
    
    def _create_nyquist_plot(self, G_s, s, omega):
        """Erstelle Nyquist-Diagramm für Komplettanalyse"""
        try:
            # Automatischer Frequenzbereich
            freq_min, freq_max = self._determine_frequency_range(G_s, s)
            omega_vals = np.logspace(np.log10(freq_min), np.log10(freq_max), 500)
            
            # G(jω) berechnen
            j = sp.I
            G_jw = G_s.subs(s, j*omega)
            
            if G_jw.is_number:
                G_vals = np.full_like(omega_vals, complex(G_jw), dtype=complex)
            else:
                G_func = sp.lambdify(omega, G_jw, 'numpy')
                G_vals = G_func(omega_vals)
            
            # Filter infinite values
            finite_mask = np.isfinite(G_vals)
            omega_clean = omega_vals[finite_mask]
            G_clean = G_vals[finite_mask]
            
            if len(G_clean) > 0:
                self._create_nyquist_plot_detailed(G_clean, omega_clean)
            else:
                st.error("Keine gültigen Frequenzgang-Werte")
                
        except Exception as e:
            st.error(f"Nyquist-Diagramm Fehler: {e}")
    
    def _create_root_locus(self, G_s, s):
        """Erstelle Wurzelortskurve"""
        st.info("🚧 Wurzelortskurve wird in Schritt 4 implementiert...")
    
    def _create_advanced_nyquist_plot(self, G_s, s, omega, freq_min, freq_max, 
                                    include_negative_freq, show_unit_circle, show_critical_point,
                                    show_encirclements, show_arrows, detailed_analysis):
        """Erstelle erweiterte Nyquist-Analyse"""
        
        # Frequenzvektor erstellen
        if include_negative_freq:
            omega_pos = np.logspace(np.log10(freq_min), np.log10(freq_max), 500)
            omega_neg = -omega_pos[::-1]  # Negative Frequenzen (rückwärts)
            omega_vals = np.concatenate([omega_neg, omega_pos])
        else:
            omega_vals = np.logspace(np.log10(freq_min), np.log10(freq_max), 1000)
        
        # G(jω) berechnen
        j = sp.I
        G_jw = G_s.subs(s, j*omega)
        
        # Numerische Auswertung
        if G_jw.is_number:
            G_vals = np.full_like(omega_vals, complex(G_jw), dtype=complex)
        else:
            G_func = sp.lambdify(omega, G_jw, 'numpy')
            G_vals = G_func(omega_vals)
        
        # Filter infinite/NaN values
        finite_mask = np.isfinite(G_vals)
        omega_clean = omega_vals[finite_mask]
        G_clean = G_vals[finite_mask]
        
        if len(G_clean) == 0:
            st.error("❌ Keine gültigen Nyquist-Werte gefunden!")
            return
        
        real_vals = np.real(G_clean)
        imag_vals = np.imag(G_clean)
        
        # Nyquist-Plot erstellen
        fig = go.Figure()
        
        # Hauptkurve
        if include_negative_freq:
            # Positive Frequenzen
            pos_mask = omega_clean >= 0
            fig.add_trace(go.Scatter(
                x=real_vals[pos_mask], y=imag_vals[pos_mask],
                mode='lines+markers',
                name='G(jω) (ω≥0)',
                line=dict(color='blue', width=2),
                marker=dict(size=2),
                hovertemplate='ω: %{text}<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>',
                text=[f'{w:.2f}' for w in omega_clean[pos_mask]]
            ))
            
            # Negative Frequenzen
            neg_mask = omega_clean < 0
            fig.add_trace(go.Scatter(
                x=real_vals[neg_mask], y=imag_vals[neg_mask],
                mode='lines+markers',
                name='G(jω) (ω<0)',
                line=dict(color='orange', width=2, dash='dash'),
                marker=dict(size=2),
                hovertemplate='ω: %{text}<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>',
                text=[f'{w:.2f}' for w in omega_clean[neg_mask]]
            ))
        else:
            fig.add_trace(go.Scatter(
                x=real_vals, y=imag_vals,
                mode='lines+markers',
                name='G(jω)',
                line=dict(color='blue', width=2),
                marker=dict(size=3),
                hovertemplate='ω: %{text}<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>',
                text=[f'{w:.2f}' for w in omega_clean]
            ))
        
        # Kritischer Punkt
        if show_critical_point:
            fig.add_trace(go.Scatter(
                x=[-1], y=[0],
                mode='markers',
                marker=dict(symbol='x', size=15, color='red', line=dict(width=3)),
                name='Kritischer Punkt (-1+0j)',
                hovertemplate='Kritischer Punkt<br>Real: -1<br>Imag: 0<extra></extra>'
            ))
        
        # Einheitskreis
        if show_unit_circle:
            theta = np.linspace(0, 2*np.pi, 100)
            unit_circle_x = np.cos(theta)
            unit_circle_y = np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=unit_circle_x, y=unit_circle_y,
                mode='lines',
                line=dict(color='lightgray', width=1, dash='dot'),
                name='Einheitskreis',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Richtungspfeile
        if show_arrows and len(omega_clean) > 10:
            # Alle 10% der Punkte einen Pfeil
            arrow_indices = np.linspace(0, len(real_vals)-1, 10, dtype=int)
            for i in arrow_indices[:-1]:
                if i < len(real_vals) - 1:
                    # Richtungsvektor
                    dx = real_vals[i+1] - real_vals[i]
                    dy = imag_vals[i+1] - imag_vals[i]
                    # Normalisieren
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        dx /= length * 20  # Skalierung
                        dy /= length * 20
                        
                        fig.add_annotation(
                            x=real_vals[i], y=imag_vals[i],
                            ax=real_vals[i] + dx, ay=imag_vals[i] + dy,
                            xref='x', yref='y',
                            axref='x', ayref='y',
                            arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='darkblue'
                        )
        
        # Koordinatenachsen
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.3)
        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.3)
        
        # Auto-Skalierung
        x_range = [min(real_vals), max(real_vals)]
        y_range = [min(imag_vals), max(imag_vals)]
        
        # Erweitere Bereich um kritischen Punkt falls nötig
        if show_critical_point:
            x_range[0] = min(x_range[0], -1.5)
            x_range[1] = max(x_range[1], 0.5)
        
        # Padding hinzufügen
        x_padding = (x_range[1] - x_range[0]) * 0.1
        y_padding = (y_range[1] - y_range[0]) * 0.1
        
        fig.update_layout(
            title="Erweiterte Nyquist-Analyse",
            xaxis_title="Realteil",
            yaxis_title="Imaginärteil",
            showlegend=True,
            width=800, height=700,
            xaxis=dict(
                range=[x_range[0]-x_padding, x_range[1]+x_padding],
                showgrid=True,
                scaleanchor="y", 
                scaleratio=1
            ),
            yaxis=dict(
                range=[y_range[0]-y_padding, y_range[1]+y_padding],
                showgrid=True
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Erweiterte Nyquist-Stabilitätsanalyse
        if show_encirclements:
            self._advanced_nyquist_stability_analysis(real_vals, imag_vals, omega_clean, G_s, s)
        
        if detailed_analysis:
            self._detailed_nyquist_analysis(real_vals, imag_vals, omega_clean)
    
    def _create_advanced_root_locus_plot(self, G_s, s, k_min, k_max, k_points,
                                       show_poles_zeros, show_asymptotes, show_breakpoints,
                                       show_k_values, interactive_k, stability_regions):
        """Erstelle erweiterte Wurzelortskurven-Analyse"""
        
        try:
            # Extrahiere Zähler und Nenner
            num = sp.numer(G_s)
            den = sp.denom(G_s)
            
            # Berechne Pole und Nullstellen
            poles = [complex(pole.evalf()) for pole in sp.solve(den, s)]
            zeros = [complex(zero.evalf()) for zero in sp.solve(num, s)]
            
            # K-Werte
            k_vals = np.linspace(k_min, k_max, k_points)
            
            # Berechne Wurzelortskurve
            root_locus_data = []
            
            for k in k_vals:
                # Charakteristische Gleichung: 1 + k*G(s) = 0
                char_eq = den + k * num
                roots = sp.solve(char_eq, s)
                
                valid_roots = []
                for root in roots:
                    try:
                        root_val = complex(root.evalf())
                        if abs(root_val.imag) < 1e10 and abs(root_val.real) < 1e10:  # Numerische Stabilität
                            valid_roots.append(root_val)
                    except:
                        continue
                
                root_locus_data.append((k, valid_roots))
            
            # Plot erstellen
            fig = go.Figure()
            
            # Pole und Nullstellen des offenen Kreises
            if show_poles_zeros:
                if poles:
                    pole_real = [p.real for p in poles]
                    pole_imag = [p.imag for p in poles]
                    fig.add_trace(go.Scatter(
                        x=pole_real, y=pole_imag,
                        mode='markers',
                        marker=dict(symbol='x', size=15, color='red', line=dict(width=3)),
                        name='Pole (offener Kreis)',
                        hovertemplate='Pol<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>'
                    ))
                
                if zeros:
                    zero_real = [z.real for z in zeros]
                    zero_imag = [z.imag for z in zeros]
                    fig.add_trace(go.Scatter(
                        x=zero_real, y=zero_imag,
                        mode='markers',
                        marker=dict(symbol='circle-open', size=12, color='blue', line=dict(width=3)),
                        name='Nullstellen',
                        hovertemplate='Nullstelle<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>'
                    ))
            
            # Wurzelortskurve zeichnen
            for i in range(len(poles)):
                branch_real = []
                branch_imag = []
                branch_k = []
                
                for k, roots in root_locus_data:
                    if i < len(roots):
                        branch_real.append(roots[i].real)
                        branch_imag.append(roots[i].imag)
                        branch_k.append(k)
                
                if branch_real:
                    hover_text = [f'K: {k:.2f}' for k in branch_k] if show_k_values else None
                    
                    fig.add_trace(go.Scatter(
                        x=branch_real, y=branch_imag,
                        mode='lines+markers',
                        line=dict(color=f'hsl({i*360/len(poles)}, 70%, 50%)', width=2),
                        marker=dict(size=3),
                        name=f'Ast {i+1}',
                        text=hover_text,
                        hovertemplate='Real: %{x:.3f}<br>Imag: %{y:.3f}<br>%{text}<extra></extra>' if show_k_values else 'Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>'
                    ))
            
            # Stabilitätsregionen
            if stability_regions:
                # Linke Halbebene (stabile Region) markieren
                # Berechne Bereich aus Daten
                all_real = []
                all_imag = []
                for k, roots in root_locus_data:
                    for root in roots:
                        all_real.append(root.real)
                        all_imag.append(root.imag)
                
                if all_real and all_imag:
                    x_min, x_max = min(all_real) - 1, max(all_real) + 1
                    y_min, y_max = min(all_imag) - 1, max(all_imag) + 1
                else:
                    x_min, x_max, y_min, y_max = -10, 10, -10, 10
                
                if all_real:
                    x_min = min(all_real) - 1
                    x_max = max(all_real) + 1
                    y_min = min(all_imag) - 1
                    y_max = max(all_imag) + 1
                
                fig.add_shape(
                    type="rect",
                    x0=x_min, y0=y_min, x1=0, y1=y_max,
                    fillcolor="lightgreen", opacity=0.2,
                    layer="below", line_width=0,
                )
                
                fig.add_shape(
                    type="rect",
                    x0=0, y0=y_min, x1=x_max, y1=y_max,
                    fillcolor="lightcoral", opacity=0.2,
                    layer="below", line_width=0,
                )
            
            # Koordinatenachsen
            fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
            fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
            
            fig.update_layout(
                title="Wurzelortskurve",
                xaxis_title="Realteil",
                yaxis_title="Imaginärteil",
                showlegend=True,
                width=800, height=700,
                xaxis=dict(showgrid=True, zeroline=True),
                yaxis=dict(showgrid=True, zeroline=True, scaleanchor="x", scaleratio=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interaktive K-Variation
            if interactive_k:
                st.markdown("### 🎛️ Interaktive K-Variation")
                k_interactive = st.slider(
                    "Verstärkung K:", 
                    min_value=float(k_min), 
                    max_value=float(k_max), 
                    value=float((k_min + k_max) / 2),
                    step=float((k_max - k_min) / 100),
                    key='k_interactive_slider'
                )
                
                # Berechne Pole für aktuelles K
                char_eq_k = den + k_interactive * num
                roots_k = sp.solve(char_eq_k, s)
                
                col_k1, col_k2 = st.columns(2)
                
                with col_k1:
                    st.markdown(f"**Pole für K = {k_interactive:.2f}:**")
                    stable = True
                    for i, root in enumerate(roots_k):
                        try:
                            root_val = complex(root.evalf())
                            real_part = root_val.real
                            imag_part = root_val.imag
                            
                            if real_part >= 0:
                                stable = False
                                st.write(f"p{i+1} = {real_part:.3f} + {imag_part:.3f}j ⚠️")
                            else:
                                st.write(f"p{i+1} = {real_part:.3f} + {imag_part:.3f}j ✅")
                        except:
                            st.write(f"p{i+1} = Numerischer Fehler")
                
                with col_k2:
                    if stable:
                        st.success("✅ **System ist stabil**")
                        st.write("Alle Pole in linker Halbebene")
                    else:
                        st.error("❌ **System ist instabil**")
                        st.write("Pole in rechter Halbebene vorhanden")
            
            # Zusätzliche Analyse
            self._root_locus_additional_analysis(poles, zeros, k_vals, root_locus_data)
            
        except Exception as e:
            st.error(f"Fehler bei Wurzelortskurven-Berechnung: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    def _analyze_stability(self, G_s, s):
        """Analysiere Stabilität"""
        st.info("🚧 Stabilitätsanalyse wird in Schritt 5 implementiert...")
    
    def _perform_comprehensive_stability_analysis(self, G_s, s, omega,
                                                show_pole_analysis, show_routh_hurwitz, show_nyquist_criterion,
                                                show_gain_margin, show_phase_margin, show_sensitivity,
                                                show_robustness, show_performance, show_recommendations):
        """Führe umfassende Stabilitätsanalyse durch"""
        
        # 1. Pol-Analyse
        if show_pole_analysis:
            st.markdown("---")
            st.markdown("### 📍 Pol-basierte Stabilitätsanalyse")
            self._pole_stability_analysis(G_s, s)
        
        # 2. Routh-Hurwitz Kriterium  
        if show_routh_hurwitz:
            st.markdown("---")
            st.markdown("### 📋 Routh-Hurwitz Kriterium")
            self._routh_hurwitz_analysis(G_s, s)
        
        # 3. Nyquist-Kriterium
        if show_nyquist_criterion:
            st.markdown("---")
            st.markdown("### 🔄 Nyquist-Stabilitätskriterium")
            self._nyquist_criterion_analysis(G_s, s, omega)
        
        # 4. Stabilitätsreserven
        if show_gain_margin or show_phase_margin:
            st.markdown("---")
            st.markdown("### 📏 Stabilitätsreserven")
            self._stability_margins_analysis(G_s, s, omega, show_gain_margin, show_phase_margin)
        
        # 5. Sensitivitätsanalyse
        if show_sensitivity:
            st.markdown("---")
            st.markdown("### 🎯 Sensitivitätsanalyse")
            self._sensitivity_analysis(G_s, s)
        
        # 6. Robustheitsanalyse
        if show_robustness:
            st.markdown("---")
            st.markdown("### 🛡️ Robustheitsanalyse")
            self._robustness_analysis(G_s, s, omega)
        
        # 7. Performance-Metriken
        if show_performance:
            st.markdown("---")
            st.markdown("### ⚡ Performance-Metriken")
            self._performance_metrics_analysis(G_s, s)
        
        # 8. Empfehlungen
        if show_recommendations:
            st.markdown("---")
            st.markdown("### 💡 Systemverbesserungs-Empfehlungen")
            self._stability_recommendations(G_s, s, omega)
    
    def _pole_stability_analysis(self, G_s, s):
        """Detaillierte Pol-basierte Stabilitätsanalyse"""
        
        try:
            # Pole berechnen
            poles = sp.solve(sp.denom(G_s), s)
            
            col_pole1, col_pole2 = st.columns(2)
            
            with col_pole1:
                st.markdown("#### 📊 Pol-Charakteristik")
                
                stable_poles = []
                unstable_poles = []
                marginal_poles = []
                
                for pole in poles:
                    pole_val = complex(pole.evalf())
                    real_part = pole_val.real
                    
                    if abs(real_part) < 1e-10:  # Auf imaginärer Achse
                        marginal_poles.append(pole_val)
                    elif real_part < 0:  # Linke Halbebene
                        stable_poles.append(pole_val)
                    else:  # Rechte Halbebene
                        unstable_poles.append(pole_val)
                
                st.write(f"**Stabile Pole:** {len(stable_poles)}")
                st.write(f"**Instabile Pole:** {len(unstable_poles)}")
                st.write(f"**Marginale Pole:** {len(marginal_poles)}")
                
                # Gesamtstabilität
                if len(unstable_poles) == 0 and len(marginal_poles) == 0:
                    st.success("✅ **System ist asymptotisch stabil**")
                elif len(unstable_poles) == 0 and len(marginal_poles) > 0:
                    st.warning("⚠️ **System ist marginal stabil**")
                else:
                    st.error("❌ **System ist instabil**")
            
            with col_pole2:
                st.markdown("#### 🔍 Detaillierte Pol-Information")
                
                if unstable_poles:
                    st.markdown("**⚠️ Instabile Pole:**")
                    for i, pole in enumerate(unstable_poles):
                        st.write(f"p{i+1} = {pole.real:.4f} + {pole.imag:.4f}j")
                
                if marginal_poles:
                    st.markdown("**⚖️ Marginale Pole:**")
                    for i, pole in enumerate(marginal_poles):
                        st.write(f"p{i+1} = {pole.real:.4f} + {pole.imag:.4f}j")
                
                if stable_poles:
                    # Dominanter Pol
                    dominant_pole = min(stable_poles, key=lambda p: abs(p.real))
                    st.markdown("**🎯 Dominanter Pol:**")
                    st.write(f"p_dom = {dominant_pole.real:.4f} + {dominant_pole.imag:.4f}j")
                    
                    # Zeitkonstante
                    if abs(dominant_pole.imag) < 1e-10:
                        tau = -1/dominant_pole.real
                        st.write(f"Zeitkonstante: τ = {tau:.3f} s")
                    else:
                        omega_n = abs(dominant_pole)
                        zeta = -dominant_pole.real / omega_n
                        st.write(f"ωₙ = {omega_n:.3f} rad/s, ζ = {zeta:.3f}")
        
        except Exception as e:
            st.error(f"Fehler bei Pol-Analyse: {e}")
    
    def _routh_hurwitz_analysis(self, G_s, s):
        """Routh-Hurwitz Stabilitätskriterium"""
        
        try:
            # Charakteristisches Polynom extrahieren
            char_poly = sp.denom(G_s)
            coeffs = sp.Poly(char_poly, s).all_coeffs()
            
            # Konvertiere zu float
            coeffs_float = [float(coeff) for coeff in coeffs]
            n = len(coeffs_float)
            
            col_routh1, col_routh2 = st.columns(2)
            
            with col_routh1:
                st.markdown("#### 📋 Routh-Tabelle")
                
                # Routh-Tabelle erstellen
                routh_table = self._create_routh_table(coeffs_float)
                
                # Tabelle anzeigen
                import pandas as pd
                
                max_cols = max(len(row) for row in routh_table)
                padded_table = []
                for i, row in enumerate(routh_table):
                    padded_row = row + [0] * (max_cols - len(row))
                    padded_table.append([f"s^{n-1-i}"] + [f"{val:.4f}" for val in padded_row])
                
                df = pd.DataFrame(padded_table, columns=['Potenz'] + [f'Spalte {j+1}' for j in range(max_cols)])
                st.dataframe(df, use_container_width=True)
            
            with col_routh2:
                st.markdown("#### 🔍 Routh-Kriterium Auswertung")
                
                # Vorzeichenwechsel in erster Spalte prüfen
                first_column = [row[0] for row in routh_table if len(row) > 0]
                sign_changes = 0
                
                for i in range(len(first_column) - 1):
                    if first_column[i] * first_column[i+1] < 0:
                        sign_changes += 1
                
                st.write(f"**Vorzeichenwechsel in 1. Spalte:** {sign_changes}")
                
                if sign_changes == 0:
                    st.success("✅ **System ist stabil** (Routh-Kriterium)")
                    st.write("Keine Vorzeichenwechsel → Alle Pole in linker Halbebene")
                else:
                    st.error(f"❌ **System ist instabil** (Routh-Kriterium)")
                    st.write(f"{sign_changes} Pole in rechter Halbebene")
                
                # Zusätzliche Bedingungen prüfen
                if any(coeff <= 0 for coeff in coeffs_float):
                    st.warning("⚠️ **Notwendige Bedingung verletzt**")
                    st.write("Nicht alle Koeffizienten positiv")
        
        except Exception as e:
            st.error(f"Fehler bei Routh-Hurwitz Analyse: {e}")
    
    def _create_routh_table(self, coeffs):
        """Erstelle Routh-Tabelle"""
        n = len(coeffs)
        
        if n < 2:
            return [[coeffs[0] if coeffs else 0]]
        
        # Initialize Routh table
        routh_table = []
        
        # First two rows
        row1 = [coeffs[i] for i in range(0, n, 2)]
        row2 = [coeffs[i] for i in range(1, n, 2)]
        
        routh_table.append(row1)
        if len(row2) > 0:  # Sicherstellen dass row2 nicht leer ist
            routh_table.append(row2)
        
        # Calculate remaining rows
        for i in range(2, n):
            new_row = []
            prev_row = routh_table[i-1] if i-1 < len(routh_table) else []
            prev_prev_row = routh_table[i-2] if i-2 < len(routh_table) else []
            
            if not prev_row or len(prev_row) == 0:
                break
                
            for j in range(max(len(prev_prev_row) - 1, 1)):
                if len(prev_row) > 0 and prev_row[0] != 0:
                    prev_prev_val = prev_prev_row[j+1] if j+1 < len(prev_prev_row) else 0
                    prev_val = prev_row[j+1] if j+1 < len(prev_row) else 0
                    val = (prev_row[0] * prev_prev_val - prev_prev_row[0] * prev_val) / prev_row[0]
                    new_row.append(val)
                else:
                    # Spezialfall: Erste Spalte ist Null
                    if j == 0:
                        new_row.append(1e-6)  # Kleiner Wert statt Null
                    else:
                        new_row.append(0)
            
            if not new_row:
                break
                
            routh_table.append(new_row)
            
            # Stoppe wenn alle Werte Null sind
            if all(abs(val) < 1e-10 for val in new_row):
                break
        
        return routh_table
    
    def _nyquist_criterion_analysis(self, G_s, s, omega):
        """Nyquist-Stabilitätskriterium Analyse"""
        
        try:
            col_nyq1, col_nyq2 = st.columns(2)
            
            with col_nyq1:
                st.markdown("#### 🔄 Nyquist-Kriterium")
                
                # Berechne Frequenzgang
                freq_min, freq_max = self._determine_frequency_range(G_s, s)
                omega_vals = np.logspace(np.log10(freq_min), np.log10(freq_max), 1000)
                
                j = sp.I
                G_jw = G_s.subs(s, j*omega)
                
                if G_jw.is_number:
                    G_vals = np.full_like(omega_vals, complex(G_jw), dtype=complex)
                else:
                    G_func = sp.lambdify(omega, G_jw, 'numpy')
                    G_vals = G_func(omega_vals)
                
                # Filter finite values
                finite_mask = np.isfinite(G_vals)
                G_clean = G_vals[finite_mask]
                
                if len(G_clean) > 0:
                    # Umkreisungsanalyse
                    real_vals = np.real(G_clean)
                    imag_vals = np.imag(G_clean)
                    
                    vectors_to_critical = np.column_stack([real_vals + 1, imag_vals])
                    angles = np.arctan2(vectors_to_critical[:, 1], vectors_to_critical[:, 0])
                    angles_unwrapped = np.unwrap(angles)
                    total_angle_change = angles_unwrapped[-1] - angles_unwrapped[0]
                    encirclements = total_angle_change / (2 * np.pi)
                    
                    st.write(f"**Umkreisungen von (-1+0j):** {encirclements:.1f}")
                    
                    # Pole in rechter Halbebene zählen
                    poles = sp.solve(sp.denom(G_s), s)
                    P = sum(1 for pole in poles if complex(pole.evalf()).real > 0)
                    
                    st.write(f"**Pole in rechter Halbebene (P):** {P}")
                    
                    # Nyquist-Kriterium: Z = N + P (Z = Nullstellen von 1+GH in RHP)
                    Z = int(round(encirclements)) + P
                    st.write(f"**Instabile Pole des geschl. Kreises (Z):** {Z}")
            
            with col_nyq2:
                st.markdown("#### 📊 Stabilitätsbewertung")
                
                if Z == 0:
                    st.success("✅ **System ist stabil** (Nyquist-Kriterium)")
                    st.write("Keine instabilen Pole im geschlossenen Kreis")
                else:
                    st.error(f"❌ **System ist instabil** (Nyquist-Kriterium)")
                    st.write(f"{Z} instabile Pole im geschlossenen Kreis")
                
                # Minimaler Abstand zum kritischen Punkt
                if len(G_clean) > 0:
                    distances = np.sqrt((real_vals + 1)**2 + imag_vals**2)
                    min_distance = np.min(distances)
                    
                    st.write(f"**Min. Abstand zu (-1+0j):** {min_distance:.3f}")
                    
                    if min_distance < 0.5:
                        st.warning("⚠️ Sehr nahe am kritischen Punkt!")
                    elif min_distance < 1.0:
                        st.info("ℹ️ Moderate Nähe zum kritischen Punkt")
                    else:
                        st.success("✅ Sicherer Abstand zum kritischen Punkt")
        
        except Exception as e:
            st.error(f"Fehler bei Nyquist-Kriterium: {e}")
    
    def _stability_margins_analysis(self, G_s, s, omega, show_gain_margin, show_phase_margin):
        """Detaillierte Stabilitätsreserven-Analyse"""
        
        try:
            # Frequenzgang berechnen
            freq_min, freq_max = self._determine_frequency_range(G_s, s)
            omega_vals = np.logspace(np.log10(freq_min), np.log10(freq_max), 1000)
            
            j = sp.I
            G_jw = G_s.subs(s, j*omega)
            
            if G_jw.is_number:
                G_vals = np.full_like(omega_vals, complex(G_jw), dtype=complex)
            else:
                G_func = sp.lambdify(omega, G_jw, 'numpy')
                G_vals = G_func(omega_vals)
            
            finite_mask = np.isfinite(G_vals)
            omega_clean = omega_vals[finite_mask]
            G_clean = G_vals[finite_mask]
            
            if len(G_clean) == 0:
                st.error("Keine gültigen Frequenzgang-Werte für Reserven-Analyse")
                return
            
            magnitude_db = 20 * np.log10(np.abs(G_clean))
            phase_deg = np.angle(G_clean) * 180 / np.pi
            
            col_margin1, col_margin2 = st.columns(2)
            
            if show_gain_margin:
                with col_margin1:
                    st.markdown("#### 📊 Amplitudenreserve (Gain Margin)")
                    
                    # Finde -180° Durchgang
                    phase_shifted = phase_deg + 180
                    phase_crossings = np.where(np.diff(np.signbit(phase_shifted)))[0]
                    
                    if len(phase_crossings) > 0:
                        idx = phase_crossings[0]
                        if idx < len(omega_clean) - 1:
                            phase_crossover_freq = np.interp(0, [phase_shifted[idx], phase_shifted[idx+1]], 
                                                           [omega_clean[idx], omega_clean[idx+1]])
                            freq_idx = np.argmin(np.abs(omega_clean - phase_crossover_freq))
                            gain_margin = -magnitude_db[freq_idx]
                            
                            st.metric("Amplitudenreserve", f"{gain_margin:.1f} dB")
                            st.write(f"Bei ω = {phase_crossover_freq:.3f} rad/s")
                            
                            # Bewertung
                            if gain_margin > 12:
                                st.success("✅ Sehr gut (> 12 dB)")
                            elif gain_margin > 6:
                                st.success("✅ Gut (6-12 dB)")
                            elif gain_margin > 3:
                                st.warning("⚠️ Akzeptabel (3-6 dB)")
                            else:
                                st.error("❌ Kritisch (< 3 dB)")
                        else:
                            st.write("Nicht berechenbar")
                    else:
                        st.info("Kein -180° Durchgang gefunden")
            
            if show_phase_margin:
                with col_margin2:
                    st.markdown("#### 📐 Phasenreserve (Phase Margin)")
                    
                    # Finde 0 dB Durchgang
                    zero_crossings = np.where(np.diff(np.signbit(magnitude_db)))[0]
                    if len(zero_crossings) > 0:
                        idx = zero_crossings[0]
                        if idx < len(omega_clean) - 1:
                            gain_crossover_freq = np.interp(0, [magnitude_db[idx], magnitude_db[idx+1]], 
                                                          [omega_clean[idx], omega_clean[idx+1]])
                            freq_idx = np.argmin(np.abs(omega_clean - gain_crossover_freq))
                            phase_at_crossover = phase_deg[freq_idx]
                            phase_margin = 180 + phase_at_crossover
                            
                            st.metric("Phasenreserve", f"{phase_margin:.1f}°")
                            st.write(f"Bei ω = {gain_crossover_freq:.3f} rad/s")
                            
                            # Bewertung
                            if phase_margin > 60:
                                st.success("✅ Sehr gut (> 60°)")
                            elif phase_margin > 45:
                                st.success("✅ Gut (45-60°)")
                            elif phase_margin > 30:
                                st.warning("⚠️ Akzeptabel (30-45°)")
                            else:
                                st.error("❌ Kritisch (< 30°)")
                        else:
                            st.write("Nicht berechenbar")
                    else:
                        st.info("Kein 0 dB Durchgang gefunden")
        
        except Exception as e:
            st.error(f"Fehler bei Stabilitätsreserven-Analyse: {e}")
    
    def _sensitivity_analysis(self, G_s, s):
        """Sensitivitätsanalyse"""
        
        st.markdown("#### 🎯 Parametersensitivität")
        
        try:
            # Sensitivitätsfunktion S = 1/(1+G)
            S = 1 / (1 + G_s)
            S_simplified = sp.simplify(S)
            
            col_sens1, col_sens2 = st.columns(2)
            
            with col_sens1:
                st.markdown("**Sensitivitätsfunktion:**")
                st.latex(f"S(s) = \\frac{{1}}{{1 + G(s)}} = {sp.latex(S_simplified)}")
                
                # Pole der Sensitivitätsfunktion
                sens_poles = sp.solve(sp.denom(S_simplified), s)
                st.markdown("**Pole der Sensitivitätsfunktion:**")
                for i, pole in enumerate(sens_poles):
                    pole_val = complex(pole.evalf())
                    st.write(f"p{i+1} = {pole_val:.4f}")
            
            with col_sens2:
                # Komplementäre Sensitivitätsfunktion T = G/(1+G)
                T = G_s / (1 + G_s)
                T_simplified = sp.simplify(T)
                
                st.markdown("**Komplementäre Sensitivitätsfunktion:**")
                st.latex(f"T(s) = \\frac{{G(s)}}{{1 + G(s)}} = {sp.latex(T_simplified)}")
                
                # Verifikation: S + T = 1
                verification = sp.simplify(S + T)
                st.write(f"Verifikation S + T = {verification}")
        
        except Exception as e:
            st.error(f"Fehler bei Sensitivitätsanalyse: {e}")
    
    def _robustness_analysis(self, G_s, s, omega):
        """Robustheitsanalyse"""
        
        st.markdown("#### 🛡️ Robustheit gegen Parameterunsicherheiten")
        
        try:
            col_rob1, col_rob2 = st.columns(2)
            
            with col_rob1:
                st.markdown("**Modellierungsunsicherheiten:**")
                
                # Multiplicative uncertainty analysis
                st.write("• **Multiplikative Unsicherheit:** ΔG = G₀ · Δ")
                st.write("• **Additive Unsicherheit:** G = G₀ + Δ")
                
                # Robuste Stabilität über kleine Verstärkungsänderung
                st.markdown("**Verstärkungs-Robustheit:**")
                
                # Frequenzgang für Robustheitsanalyse
                freq_min, freq_max = self._determine_frequency_range(G_s, s)
                omega_vals = np.logspace(np.log10(freq_min), np.log10(freq_max), 500)
                
                j = sp.I
                G_jw = G_s.subs(s, j*omega)
                
                if not G_jw.is_number:
                    G_func = sp.lambdify(omega, G_jw, 'numpy')
                    G_vals = G_func(omega_vals)
                    
                    finite_mask = np.isfinite(G_vals)
                    G_clean = G_vals[finite_mask]
                    
                    if len(G_clean) > 0:
                        # Robuste Stabilität - minimaler Abstand zu kritischem Punkt
                        distances = np.abs(G_clean + 1)
                        min_distance = np.min(distances)
                        
                        st.write(f"Min. Abstand zu (-1,0): {min_distance:.3f}")
                        
                        # Verstärkungstoleranz
                        if min_distance > 0:
                            gain_tolerance = 20 * np.log10(min_distance)
                            st.write(f"Verstärkungstoleranz: ±{gain_tolerance:.1f} dB")
            
            with col_rob2:
                st.markdown("**Performance-Robustheit:**")
                
                # Pole der geschlossenen Schleife
                closed_loop_den = sp.denom(G_s) + sp.numer(G_s)
                cl_poles = sp.solve(closed_loop_den, s)
                
                # Dämpfungsanalyse
                min_damping = float('inf')
                for pole in cl_poles:
                    pole_val = complex(pole.evalf())
                    if abs(pole_val.imag) > 1e-10:  # Komplexer Pol
                        omega_n = abs(pole_val)
                        zeta = -pole_val.real / omega_n
                        min_damping = min(min_damping, zeta)
                
                if min_damping != float('inf'):
                    st.write(f"Minimale Dämpfung: ζ = {min_damping:.3f}")
                    
                    if min_damping > 0.7:
                        st.success("✅ Sehr gut gedämpft")
                    elif min_damping > 0.5:
                        st.success("✅ Gut gedämpft")
                    elif min_damping > 0.3:
                        st.warning("⚠️ Mäßig gedämpft")
                    else:
                        st.error("❌ Schlecht gedämpft")
                else:
                    st.info("Nur reelle Pole - keine Schwingungen")
        
        except Exception as e:
            st.error(f"Fehler bei Robustheitsanalyse: {e}")
    
    def _performance_metrics_analysis(self, G_s, s):
        """Performance-Metriken Analyse"""
        
        st.markdown("#### ⚡ Zeitbereich Performance-Metriken")
        
        try:
            # Geschlossener Kreis
            T = G_s / (1 + G_s)
            T_simplified = sp.simplify(T)
            
            col_perf1, col_perf2 = st.columns(2)
            
            with col_perf1:
                st.markdown("**Übertragungsfunktion geschlossener Kreis:**")
                st.latex(f"T(s) = {sp.latex(T_simplified)}")
                
                # Pole des geschlossenen Kreises
                cl_poles = sp.solve(sp.denom(T_simplified), s)
                st.markdown("**Pole des geschlossenen Kreises:**")
                
                dominant_pole = None
                min_real = 0
                
                for i, pole in enumerate(cl_poles):
                    pole_val = complex(pole.evalf())
                    st.write(f"p{i+1} = {pole_val:.4f}")
                    
                    if pole_val.real < 0 and pole_val.real > min_real:
                        min_real = pole_val.real
                        dominant_pole = pole_val
            
            with col_perf2:
                st.markdown("**Zeitbereich-Charakteristik:**")
                
                if dominant_pole is not None:
                    if abs(dominant_pole.imag) < 1e-10:  # Reeller dominanter Pol
                        tau = -1/dominant_pole.real
                        st.write(f"Zeitkonstante: τ = {tau:.3f} s")
                        st.write(f"Anstiegszeit: tr ≈ {2.2*tau:.3f} s")
                        st.write(f"Einschwingzeit: ts ≈ {4*tau:.3f} s")
                        st.write("Überschwingen: 0% (aperiodisch)")
                        
                    else:  # Komplexer dominanter Pol
                        omega_n = abs(dominant_pole)
                        zeta = -dominant_pole.real / omega_n
                        omega_d = omega_n * np.sqrt(1 - zeta**2)
                        
                        st.write(f"Eigenfrequenz: ωₙ = {omega_n:.3f} rad/s")
                        st.write(f"Dämpfung: ζ = {zeta:.3f}")
                        st.write(f"Gedämpfte Frequenz: ωd = {omega_d:.3f} rad/s")
                        
                        if zeta < 1:
                            overshoot = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100
                            peak_time = np.pi / omega_d
                            settling_time = 4 / (zeta * omega_n)
                            
                            st.write(f"Überschwingen: {overshoot:.1f}%")
                            st.write(f"Ankunftszeit: tp = {peak_time:.3f} s")
                            st.write(f"Einschwingzeit: ts = {settling_time:.3f} s")
                
                # Stationärer Fehler
                try:
                    # Typ des Systems bestimmen
                    s_power = 0
                    G_expanded = sp.expand(G_s)
                    
                    # Prüfe auf Integratoren (Pole bei s=0)
                    poles = sp.solve(sp.denom(G_s), s)
                    for pole in poles:
                        if abs(complex(pole.evalf())) < 1e-10:
                            s_power += 1
                    
                    st.markdown("**Stationäre Fehler:**")
                    st.write(f"Systemtyp: {s_power}")
                    
                    if s_power == 0:
                        st.write("Sprungfehler: endlich")
                        st.write("Rampenfehler: ∞")
                    elif s_power == 1:
                        st.write("Sprungfehler: 0")
                        st.write("Rampenfehler: endlich")
                    else:
                        st.write("Sprungfehler: 0")
                        st.write("Rampenfehler: 0")
                        
                except Exception:
                    st.write("Stationäre Fehler: Berechnung nicht möglich")
        
        except Exception as e:
            st.error(f"Fehler bei Performance-Analyse: {e}")
    
    def _stability_recommendations(self, G_s, s, omega):
        """Systemverbesserungs-Empfehlungen"""
        
        st.markdown("#### 💡 Empfehlungen zur Systemverbesserung")
        
        try:
            # Analysiere aktuellen Zustand
            poles = sp.solve(sp.denom(G_s), s)
            unstable_poles = [pole for pole in poles if complex(pole.evalf()).real >= 0]
            
            # Frequenzgang-Analyse für Reserven
            freq_min, freq_max = self._determine_frequency_range(G_s, s)
            omega_vals = np.logspace(np.log10(freq_min), np.log10(freq_max), 500)
            
            j = sp.I
            G_jw = G_s.subs(s, j*omega)
            
            recommendations = []
            
            # Stabilitäts-Empfehlungen
            if unstable_poles:
                recommendations.append({
                    "category": "🚨 Kritisch",
                    "title": "System ist instabil",
                    "description": f"Das System hat {len(unstable_poles)} instabile Pole in der rechten Halbebene.",
                    "actions": [
                        "Reduzierung der Schleifenverstärkung",
                        "Hinzufügung von Kompensationsgliedern (Lead/Lag)",
                        "Verwendung von Zustandsregelung",
                        "Implementierung einer robusten Regelung"
                    ]
                })
            
            # Reserven-Empfehlungen
            if not G_jw.is_number:
                G_func = sp.lambdify(omega, G_jw, 'numpy')
                G_vals = G_func(omega_vals)
                
                finite_mask = np.isfinite(G_vals)
                G_clean = G_vals[finite_mask]
                
                if len(G_clean) > 0:
                    magnitude_db = 20 * np.log10(np.abs(G_clean))
                    phase_deg = np.angle(G_clean) * 180 / np.pi
                    
                    # Phasenreserve prüfen
                    zero_crossings = np.where(np.diff(np.signbit(magnitude_db)))[0]
                    if len(zero_crossings) > 0:
                        idx = zero_crossings[0]
                        if idx < len(omega_vals) - 1:
                            freq_idx = np.argmin(np.abs(magnitude_db))
                            phase_margin = 180 + phase_deg[freq_idx]
                            
                            if phase_margin < 45:
                                recommendations.append({
                                    "category": "⚠️ Verbesserung",
                                    "title": "Niedrige Phasenreserve",
                                    "description": f"Phasenreserve beträgt nur {phase_margin:.1f}° (< 45°)",
                                    "actions": [
                                        "Lead-Kompensation zur Phasenanhebung",
                                        "Reduzierung der Verstärkung bei kritischen Frequenzen",
                                        "Optimierung der Regler-Parameter"
                                    ]
                                })
            
            # Performance-Empfehlungen
            cl_den = sp.denom(G_s) + sp.numer(G_s)
            cl_poles = sp.solve(cl_den, s)
            
            min_damping = float('inf')
            for pole in cl_poles:
                pole_val = complex(pole.evalf())
                if abs(pole_val.imag) > 1e-10:
                    omega_n = abs(pole_val)
                    zeta = -pole_val.real / omega_n
                    min_damping = min(min_damping, zeta)
            
            if min_damping != float('inf') and min_damping < 0.5:
                recommendations.append({
                    "category": "🎯 Performance",
                    "title": "Geringe Dämpfung",
                    "description": f"Minimale Dämpfung beträgt ζ = {min_damping:.3f} (< 0.5)",
                    "actions": [
                        "Erhöhung der Dämpfung durch Lag-Kompensation",
                        "Notch-Filter bei Resonanzfrequenzen",
                        "Anpassung der Regler-Struktur"
                    ]
                })
            
            # Positive Empfehlungen
            if not unstable_poles and min_damping > 0.5:
                recommendations.append({
                    "category": "✅ Gut",
                    "title": "System ist stabil und gut gedämpft",
                    "description": "Das System zeigt gute Stabilitäts- und Performance-Eigenschaften.",
                    "actions": [
                        "System kann so verwendet werden",
                        "Überwachung der Robustheit bei Parameteränderungen",
                        "Optimierung für spezielle Anforderungen möglich"
                    ]
                })
            
            # Empfehlungen anzeigen
            for rec in recommendations:
                with st.expander(f"{rec['category']}: {rec['title']}"):
                    st.write(rec['description'])
                    st.markdown("**Empfohlene Maßnahmen:**")
                    for action in rec['actions']:
                        st.write(f"• {action}")
        
        except Exception as e:
            st.error(f"Fehler bei Empfehlungs-Generierung: {e}")
    
    # Hilfsmethoden für erweiterte Analysen
    def _advanced_nyquist_stability_analysis(self, real_vals, imag_vals, omega_vals, G_s, s):
        """Erweiterte Nyquist-Stabilitätsanalyse"""
        
        st.markdown("#### 🔍 Erweiterte Nyquist-Stabilitätsanalyse")
        
        try:
            col_nyq_adv1, col_nyq_adv2 = st.columns(2)
            
            with col_nyq_adv1:
                # Detaillierte Umkreisungsanalyse
                st.markdown("**Umkreisungsanalyse:**")
                
                vectors_to_critical = np.column_stack([real_vals + 1, imag_vals])
                angles = np.arctan2(vectors_to_critical[:, 1], vectors_to_critical[:, 0])
                angles_unwrapped = np.unwrap(angles)
                total_angle_change = angles_unwrapped[-1] - angles_unwrapped[0]
                encirclements = total_angle_change / (2 * np.pi)
                
                st.write(f"Gesamte Winkeländerung: {total_angle_change:.2f} rad")
                st.write(f"Umkreisungen: N = {encirclements:.2f}")
                
                # Pole in RHP
                poles = sp.solve(sp.denom(G_s), s)
                P = sum(1 for pole in poles if complex(pole.evalf()).real > 0)
                st.write(f"Pole in RHP: P = {P}")
                
                # Stabilität nach Nyquist
                Z = int(round(encirclements)) + P
                st.write(f"Instabile Pole (geschl. Kreis): Z = {Z}")
                
                if Z == 0:
                    st.success("✅ Stabil nach Nyquist-Kriterium")
                else:
                    st.error(f"❌ Instabil: {Z} Pole in RHP")
            
            with col_nyq_adv2:
                # Kritische Frequenzen
                st.markdown("**Kritische Punkte:**")
                
                # Minimaler Abstand
                distances = np.sqrt((real_vals + 1)**2 + imag_vals**2)
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                critical_freq = omega_vals[min_idx]
                
                st.write(f"Min. Abstand: {min_distance:.4f}")
                st.write(f"Bei ω = {critical_freq:.3f} rad/s")
                
                # Nähe-Bewertung
                if min_distance < 0.1:
                    st.error("❌ Extrem nah - sehr kritisch!")
                elif min_distance < 0.5:
                    st.warning("⚠️ Sehr nah - kritisch")
                elif min_distance < 1.0:
                    st.warning("⚠️ Nah - beobachten")
                else:
                    st.success("✅ Sicherer Abstand")
                
                # Kreuzungspunkte mit reeller Achse
                real_axis_crossings = []
                for i in range(len(imag_vals) - 1):
                    if imag_vals[i] * imag_vals[i+1] < 0:  # Vorzeichenwechsel
                        crossing_real = np.interp(0, [imag_vals[i], imag_vals[i+1]], 
                                                [real_vals[i], real_vals[i+1]])
                        crossing_freq = np.interp(0, [imag_vals[i], imag_vals[i+1]], 
                                                [omega_vals[i], omega_vals[i+1]])
                        real_axis_crossings.append((crossing_real, crossing_freq))
                
                if real_axis_crossings:
                    st.markdown("**Kreuzungen mit reeller Achse:**")
                    for i, (crossing_real, freq) in enumerate(real_axis_crossings[:3]):
                        st.write(f"{i+1}: Re = {crossing_real:.3f} bei ω = {freq:.3f}")
        
        except Exception as e:
            st.error(f"Fehler bei erweiterter Nyquist-Analyse: {e}")
    
    def _detailed_nyquist_analysis(self, real_vals, imag_vals, omega_vals):
        """Detaillierte Nyquist-Kurven-Analyse"""
        
        st.markdown("#### 📊 Detaillierte Kurvenanalyse")
        
        try:
            col_detail1, col_detail2 = st.columns(2)
            
            with col_detail1:
                st.markdown("**Kurven-Charakteristik:**")
                
                # Maximale und minimale Werte
                max_real = np.max(real_vals)
                min_real = np.min(real_vals)
                max_imag = np.max(imag_vals)
                min_imag = np.min(imag_vals)
                
                st.write(f"Real: [{min_real:.3f}, {max_real:.3f}]")
                st.write(f"Imag: [{min_imag:.3f}, {max_imag:.3f}]")
                
                # Kurven-Länge
                distances = np.sqrt(np.diff(real_vals)**2 + np.diff(imag_vals)**2)
                total_length = np.sum(distances)
                st.write(f"Kurvenlänge: {total_length:.3f}")
                
                # Wendepunkte
                curvature = np.abs(np.diff(np.diff(real_vals)) + 1j*np.diff(np.diff(imag_vals)))
                high_curvature_indices = np.where(curvature > np.percentile(curvature, 90))[0]
                st.write(f"Wendepunkte: {len(high_curvature_indices)}")
            
            with col_detail2:
                st.markdown("**Frequenz-Charakteristik:**")
                
                # Niedrige und hohe Frequenzen
                low_freq_idx = omega_vals < np.percentile(omega_vals, 10)
                high_freq_idx = omega_vals > np.percentile(omega_vals, 90)
                
                if np.any(low_freq_idx):
                    low_freq_center = (np.mean(real_vals[low_freq_idx]), 
                                     np.mean(imag_vals[low_freq_idx]))
                    st.write(f"Niederfrequenz-Zentrum: ({low_freq_center[0]:.3f}, {low_freq_center[1]:.3f})")
                
                if np.any(high_freq_idx):
                    high_freq_center = (np.mean(real_vals[high_freq_idx]), 
                                      np.mean(imag_vals[high_freq_idx]))
                    st.write(f"Hochfrequenz-Zentrum: ({high_freq_center[0]:.3f}, {high_freq_center[1]:.3f})")
                
                # Symmetrie-Analyse
                if len(real_vals) > 10:
                    upper_half = imag_vals > 0
                    lower_half = imag_vals < 0
                    
                    if np.any(upper_half) and np.any(lower_half):
                        symmetry = np.corrcoef(real_vals[upper_half][:min(10, np.sum(upper_half))], 
                                             real_vals[lower_half][:min(10, np.sum(lower_half))])[0, 1]
                        st.write(f"Symmetrie-Index: {symmetry:.3f}")
        
        except Exception as e:
            st.error(f"Fehler bei detaillierter Analyse: {e}")
    
    def _root_locus_additional_analysis(self, poles, zeros, k_vals, root_locus_data):
        """Zusätzliche Wurzelortskurven-Analyse"""
        
        st.markdown("#### 📈 Zusätzliche Wurzelortskurven-Analyse")
        
        try:
            col_rl_add1, col_rl_add2 = st.columns(2)
            
            with col_rl_add1:
                st.markdown("**Asymptoten-Analyse:**")
                
                n_poles = len(poles)
                n_zeros = len(zeros)
                n_asymptotes = n_poles - n_zeros
                
                if n_asymptotes > 0:
                    # Asymptoten-Winkel
                    asymptote_angles = [(2*k + 1) * 180 / n_asymptotes for k in range(n_asymptotes)]
                    st.write(f"Anzahl Asymptoten: {n_asymptotes}")
                    st.write("Asymptoten-Winkel:")
                    for i, angle in enumerate(asymptote_angles):
                        st.write(f"  θ{i+1} = {angle:.1f}°")
                    
                    # Asymptoten-Zentrum
                    pole_sum = sum(poles)
                    zero_sum = sum(zeros) if zeros else 0
                    centroid = (pole_sum - zero_sum) / n_asymptotes
                    st.write(f"Asymptoten-Zentrum: {centroid:.3f}")
                else:
                    st.write("Keine Asymptoten (n_zeros ≥ n_poles)")
            
            with col_rl_add2:
                st.markdown("**Stabilitäts-Grenzwerte:**")
                
                # Finde K-Werte wo System marginal stabil wird
                marginal_k_values = []
                
                for k, roots in root_locus_data:
                    for root in roots:
                        if abs(root.real) < 1e-3:  # Nahe imaginärer Achse
                            marginal_k_values.append((k, root))
                
                if marginal_k_values:
                    st.write("Marginale Stabilität bei:")
                    for k, root in marginal_k_values[:3]:  # Zeige maximal 3
                        st.write(f"  K = {k:.3f}, ω = {abs(root.imag):.3f}")
                else:
                    st.write("Keine marginale Stabilität im K-Bereich")
                
                # Optimaler K-Bereich
                stable_k_range = []
                for k, roots in root_locus_data:
                    if all(root.real < 0 for root in roots):
                        stable_k_range.append(k)
                
                if stable_k_range:
                    st.write(f"Stabiler K-Bereich: [{min(stable_k_range):.3f}, {max(stable_k_range):.3f}]")
                else:
                    st.write("Kein stabiler K-Bereich gefunden")
        
        except Exception as e:
            st.error(f"Fehler bei zusätzlicher Wurzelortskurven-Analyse: {e}")
    
    def _calculate_stability_margins(self, G_s, s, omega):
        """Berechne Stabilitätsreserven"""
        # Diese Methode wird jetzt von _stability_margins_analysis ersetzt
        st.info("🔗 Detaillierte Stabilitätsreserven finden Sie im Stabilitäts-Tab")
