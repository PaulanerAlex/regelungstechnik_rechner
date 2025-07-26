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
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "🎯 Komplettanalyse",
                "📊 Pol-Nullstellen", 
                "📈 Frequenzgang",
                "🔄 Nyquist",
                "🌿 Wurzelortskurve",
                "⚖️ Stabilität",
                "📍 Ortskurve",
                "⚖️ Vergleichsanalyse"
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
            
            with tab8:
                self.comparison_analysis()
    
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
            show_asymptotic_complete = st.checkbox("📐 Asymptotische Bode-Näherung", value=False, key='show_asymptotic_complete')
        
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
                self._create_bode_plot(G_s, s, omega, show_asymptotic_complete)
            
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
            
            # Debug-Ausgaben
            st.markdown("##### 🔍 Debug-Informationen")
            with st.expander("System-Polynome anzeigen"):
                st.write(f"**Zählerpolynom (Numerator):** {num}")
                st.write(f"**Nennerpolynom (Denominator):** {den}")
            
            # Pole und Nullstellen berechnen - robustere Methode
            try:
                poles = sp.solve(den, s)
                if not poles:  # Falls solve() leer zurückgibt, versuche alternatives Verfahren
                    # Versuche Faktorisierung
                    den_factored = sp.factor(den)
                    st.write(f"Faktorisierter Nenner: {den_factored}")
                    
                    # Extrahiere Pole aus Faktoren
                    if den_factored != den:
                        poles = sp.solve(den_factored, s)
                    
                    # Falls immer noch leer, verwende numerische Methode
                    if not poles:
                        # Konvertiere zu numpy Polynom-Koeffizienten
                        den_poly = sp.Poly(den, s)
                        coeffs = [float(c) for c in den_poly.all_coeffs()]
                        if len(coeffs) > 1:
                            numerical_poles = np.roots(coeffs)
                            poles = [sp.sympify(complex(p)) for p in numerical_poles]
                        
            except Exception as e:
                st.error(f"Fehler bei Pol-Berechnung: {e}")
                poles = []
            
            try:
                zeros = sp.solve(num, s)
                if not zeros:  # Fallback für Nullstellen
                    num_factored = sp.factor(num)
                    if num_factored != num:
                        zeros = sp.solve(num_factored, s)
                    
                    # Numerische Methode für Nullstellen
                    if not zeros:
                        num_poly = sp.Poly(num, s)
                        coeffs = [float(c) for c in num_poly.all_coeffs()]
                        if len(coeffs) > 1:
                            numerical_zeros = np.roots(coeffs)
                            zeros = [sp.sympify(complex(z)) for z in numerical_zeros]
                            
            except Exception as e:
                st.error(f"Fehler bei Nullstellen-Berechnung: {e}")
                zeros = []
            
            # Debug-Ausgabe der gefundenen Werte
            st.write(f"**Gefundene Pole:** {len(poles)}")
            st.write(f"**Gefundene Nullstellen:** {len(zeros)}")
            
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
            
            # Pole und Nullstellen berechnen - robuste Methode
            try:
                poles = sp.solve(den, s)
                if not poles:
                    # Fallback: numerische Methode
                    den_poly = sp.Poly(den, s)
                    coeffs = [float(c) for c in den_poly.all_coeffs()]
                    if len(coeffs) > 1:
                        numerical_poles = np.roots(coeffs)
                        poles = [sp.sympify(complex(p)) for p in numerical_poles]
            except:
                poles = []
            
            try:
                zeros = sp.solve(num, s)
                if not zeros:
                    # Fallback: numerische Methode
                    num_poly = sp.Poly(num, s)
                    coeffs = [float(c) for c in num_poly.all_coeffs()]
                    if len(coeffs) > 1:
                        numerical_zeros = np.roots(coeffs)
                        zeros = [sp.sympify(complex(z)) for z in numerical_zeros]
            except:
                zeros = []
            
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
            
            st.plotly_chart(fig, use_container_width=True, key="pole_zero_plot")
            
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
            show_asymptotic = st.checkbox("📐 Asymptotische Näherung (Knick-Approximation)", value=True, key='show_asymptotic_freq')
        
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
                    self._create_bode_plot_detailed(omega_clean, magnitude_db, phase_deg, G_s, s, show_asymptotic)
                
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
                    self._analyze_stability_margins_detailed(omega_clean, magnitude_db, phase_deg, G_s, s)
                
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
            # Debug: Ausgabe der Übertragungsfunktion
            st.write(f"Debug: G(s) = {G_s}")
            
            # Charakteristische Frequenzen sammeln
            char_freqs = []
            
            # Pole extrahieren
            den = sp.denom(G_s)
            try:
                poles = sp.solve(den, s)
                st.write(f"Debug: Gefundene Pole: {poles}")
                
                for pole in poles:
                    pole_val = complex(pole.evalf())
                    # Nur Pole mit Re < 0 und |pole| > 0.001 berücksichtigen
                    if pole_val.real < 0 and abs(pole_val) > 0.001:
                        char_freqs.append(abs(pole_val))
                        st.write(f"Debug: Charakteristische Frequenz aus Pol {pole_val}: {abs(pole_val):.3f}")
                        
            except Exception as e:
                st.write(f"Debug: Pol-Extraktion fehlgeschlagen: {e}")
            
            # Nullstellen extrahieren
            num = sp.numer(G_s)
            try:
                zeros = sp.solve(num, s)
                st.write(f"Debug: Gefundene Nullstellen: {zeros}")
                
                for zero in zeros:
                    zero_val = complex(zero.evalf())
                    if abs(zero_val) > 0.001:
                        char_freqs.append(abs(zero_val))
                        st.write(f"Debug: Charakteristische Frequenz aus Nullstelle {zero_val}: {abs(zero_val):.3f}")
                        
            except Exception as e:
                st.write(f"Debug: Nullstellen-Extraktion fehlgeschlagen: {e}")
            
            # Bestimme Frequenzbereich
            if char_freqs:
                min_char_freq = min(char_freqs)
                max_char_freq = max(char_freqs)
                
                # Verbesserte Bereichsbestimmung: weniger extreme Erweiterung
                min_freq = max(min_char_freq / 10, 0.01)   # 1 Dekade darunter, min 0.01
                max_freq = min(max_char_freq * 10, 1000)   # 1 Dekade darüber, max 1000
                
                st.write(f"Debug: Char. Freqs: {char_freqs}")
                st.write(f"Debug: Bereich: {min_freq:.3f} - {max_freq:.1f} rad/s")
                
                return min_freq, max_freq
            else:
                st.write("Debug: Keine charakteristischen Frequenzen, verwende Standard")
                # Standard für Systeme ohne erkennbare charakteristische Frequenzen
                return 0.01, 100.0
                
        except Exception as e:
            st.write(f"Debug: Fehler bei Frequenzbereich-Bestimmung: {e}")
            return 0.01, 100.0
    
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
            
            st.plotly_chart(fig, use_container_width=True, key="ortskurve_plot")
            
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
    
    def _create_bode_plot_detailed(self, omega_vals, magnitude_db, phase_deg, G_s, s, show_asymptotic=False):
        """Erstelle detailliertes Bode-Diagramm mit exakter und asymptotischer Darstellung"""
        
        # Erstelle Subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Amplitudengang |G(jω)| [dB]', 'Phasengang ∠G(jω) [°]'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Exakte Kurven (Hauptdarstellung)
        fig.add_trace(
            go.Scatter(
                x=omega_vals, y=magnitude_db,
                mode='lines',
                name='|G(jω)| exakt',
                line=dict(color='blue', width=2),
                hovertemplate='ω: %{x:.3f} rad/s<br>|G|: %{y:.1f} dB<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=omega_vals, y=phase_deg,
                mode='lines',
                name='∠G(jω) exakt',
                line=dict(color='red', width=2),
                hovertemplate='ω: %{x:.3f} rad/s<br>∠G: %{y:.1f}°<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Asymptotische Näherung hinzufügen
        if show_asymptotic:
            try:
                # Berechne asymptotische Approximation
                omega_asymp, mag_asymp, phase_asymp = self._calculate_asymptotic_bode(G_s, s, omega_vals)
                
                if omega_asymp is not None and len(omega_asymp) > 0:
                    # Asymptotische Amplitude
                    fig.add_trace(
                        go.Scatter(
                            x=omega_asymp, y=mag_asymp,
                            mode='lines',
                            name='|G(jω)| asymptotisch',
                            line=dict(color='lightblue', width=2, dash='dash'),
                            hovertemplate='ω: %{x:.3f} rad/s<br>|G|_asymp: %{y:.1f} dB<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Asymptotische Phase
                    fig.add_trace(
                        go.Scatter(
                            x=omega_asymp, y=phase_asymp,
                            mode='lines',
                            name='∠G(jω) asymptotisch',
                            line=dict(color='lightcoral', width=2, dash='dash'),
                            hovertemplate='ω: %{x:.3f} rad/s<br>∠G_asymp: %{y:.1f}°<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Vergleich der Stabilitätsreserven
                    st.markdown("#### 📊 Vergleich: Exakt vs. Asymptotisch")
                    self._compare_stability_margins(omega_vals, magnitude_db, phase_deg, omega_asymp, mag_asymp, phase_asymp)
                
            except Exception as e:
                st.warning(f"⚠️ Asymptotische Approximation konnte nicht berechnet werden: {e}")
        
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
        
        # Knickfrequenz-Marker hinzufügen
        self._add_break_frequency_markers(fig, G_s, s, omega_vals, magnitude_db, phase_deg)
        
        fig.update_layout(
            title="Bode-Diagramm" + (" - Exakt vs. Asymptotisch" if show_asymptotic else ""),
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="bode_detailed_plot")
        
        # Wichtige Frequenzen markieren
        self._mark_important_frequencies(omega_vals, magnitude_db, phase_deg, G_s, s)
    
    def _calculate_asymptotic_bode(self, G_s, s, omega_vals):
        """Berechne asymptotische Bode-Approximation (Knick-Diagramm)"""
        try:
            # Extrahiere Faktoren der Übertragungsfunktion
            num = sp.numer(G_s)
            den = sp.denom(G_s)
            
            # Finde Pole und Nullstellen
            poles = sp.solve(den, s)
            zeros = sp.solve(num, s)
            
            # DC-Verstärkung
            try:
                K_dc = float(G_s.subs(s, 0))
            except:
                # Falls Pol bei s=0, verwende Grenzwert
                K_dc = 1.0
            
            # Charakteristische Frequenzen (Knickpunkte)
            break_freqs = []
            break_types = []  # 'zero' oder 'pole'
            break_orders = []  # Ordnung (Vielfachheit)
            
            # Pole verarbeiten
            for pole in poles:
                pole_val = complex(pole.evalf())
                if abs(pole_val.imag) < 1e-10:  # Reeller Pol
                    if pole_val.real < 0:  # Stabiler Pol
                        break_freq = abs(pole_val.real)
                        if break_freq > 0:
                            break_freqs.append(break_freq)
                            break_types.append('pole')
                            break_orders.append(1)
                else:  # Komplexer Pol
                    if pole_val.real < 0:  # Stabiler komplexer Pol
                        break_freq = abs(pole_val)  # Eigenfrequenz
                        if break_freq > 0:
                            break_freqs.append(break_freq)
                            break_types.append('complex_pole')
                            break_orders.append(2)  # Komplexes Paar
            
            # Nullstellen verarbeiten
            for zero in zeros:
                zero_val = complex(zero.evalf())
                if abs(zero_val.imag) < 1e-10:  # Reelle Nullstelle
                    break_freq = abs(zero_val.real)
                    if break_freq > 0:
                        break_freqs.append(break_freq)
                        break_types.append('zero')
                        break_orders.append(1)
                else:  # Komplexe Nullstelle
                    break_freq = abs(zero_val)
                    if break_freq > 0:
                        break_freqs.append(break_freq)
                        break_types.append('complex_zero')
                        break_orders.append(2)
            
            # Sortiere nach Frequenz
            sorted_data = sorted(zip(break_freqs, break_types, break_orders))
            break_freqs = [x[0] for x in sorted_data]
            break_types = [x[1] for x in sorted_data]
            break_orders = [x[2] for x in sorted_data]
            
            # Erweitere Frequenzbereich für Asymptoten
            omega_min = min(omega_vals) / 10
            omega_max = max(omega_vals) * 10
            
            # Erstelle asymptotische Frequenzpunkte
            omega_asymp_list = [omega_min]
            
            # Füge Knickpunkte hinzu
            for freq in break_freqs:
                if omega_min <= freq <= omega_max:
                    omega_asymp_list.extend([freq * 0.999, freq, freq * 1.001])
            
            omega_asymp_list.append(omega_max)
            omega_asymp = np.array(sorted(set(omega_asymp_list)))
            
            # Berechne asymptotische Amplitude
            mag_asymp = np.zeros_like(omega_asymp)
            
            # DC-Verstärkung (Startwert)
            if K_dc != 0:
                dc_gain_db = 20 * np.log10(abs(K_dc))
            else:
                dc_gain_db = 0
            
            # Systemordnung für Hochfrequenz-Asymptote
            num_degree = sp.degree(num, s) if sp.degree(num, s) is not None else 0
            den_degree = sp.degree(den, s) if sp.degree(den, s) is not None else 0
            system_order = int(den_degree) - int(num_degree)
            
            for i, omega in enumerate(omega_asymp):
                # Starte mit DC-Verstärkung
                current_gain_db = dc_gain_db
                
                # Addiere Beiträge aller Knickpunkte
                for freq, btype, order in zip(break_freqs, break_types, break_orders):
                    if omega > freq:  # Oberhalb der Knickfrequenz
                        if 'pole' in btype:
                            # Pol: -20*order dB/Dekade
                            current_gain_db -= 20 * order * np.log10(omega / freq)
                        elif 'zero' in btype:
                            # Nullstelle: +20*order dB/Dekade
                            current_gain_db += 20 * order * np.log10(omega / freq)
                
                mag_asymp[i] = current_gain_db
            
            # Berechne asymptotische Phase
            phase_asymp = np.zeros_like(omega_asymp)
            
            for i, omega in enumerate(omega_asymp):
                current_phase = 0
                
                # DC-Phasenbeitrag
                if K_dc < 0:
                    current_phase += 180
                
                # Beiträge der Knickpunkte
                for freq, btype, order in zip(break_freqs, break_types, break_orders):
                    if 'pole' in btype:
                        if 'complex' in btype:
                            # Komplexer Pol: Schneller Phasenabfall bei Knickfrequenz
                            if omega < freq / 10:
                                phase_contrib = 0
                            elif omega > freq * 10:
                                phase_contrib = -180 * order
                            else:
                                # Vereinfachte Knick-Approximation
                                phase_contrib = -90 * order * np.log10(omega / (freq / 10)) / np.log10(100)
                        else:
                            # Reeller Pol: -90°/Pol oberhalb Knickfrequenz
                            if omega > freq:
                                phase_contrib = -90 * order
                            else:
                                phase_contrib = 0
                    elif 'zero' in btype:
                        if 'complex' in btype:
                            # Komplexe Nullstelle
                            if omega < freq / 10:
                                phase_contrib = 0
                            elif omega > freq * 10:
                                phase_contrib = 180 * order
                            else:
                                phase_contrib = 90 * order * np.log10(omega / (freq / 10)) / np.log10(100)
                        else:
                            # Reelle Nullstelle: +90°/Nullstelle oberhalb Knickfrequenz
                            if omega > freq:
                                phase_contrib = 90 * order
                            else:
                                phase_contrib = 0
                    
                    current_phase += phase_contrib
                
                phase_asymp[i] = current_phase
            
            return omega_asymp, mag_asymp, phase_asymp
            
        except Exception as e:
            st.error(f"Fehler bei asymptotischer Approximation: {e}")
            return None, None, None
    
    def _compare_stability_margins(self, omega_exact, mag_exact, phase_exact, omega_asymp, mag_asymp, phase_asymp):
        """Vergleiche Stabilitätsreserven zwischen exakter und asymptotischer Berechnung"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Exakte Berechnung")
            
            # Amplitudenreserve (exakt)
            phase_shifted = phase_exact + 180
            phase_crossings = np.where(np.diff(np.signbit(phase_shifted)))[0]
            
            if len(phase_crossings) > 0:
                idx = phase_crossings[0]
                if idx < len(omega_exact) - 1:
                    phase_crossover_freq = np.interp(0, [phase_shifted[idx], phase_shifted[idx+1]], 
                                                   [omega_exact[idx], omega_exact[idx+1]])
                    freq_idx = np.argmin(np.abs(omega_exact - phase_crossover_freq))
                    gain_margin_exact = -mag_exact[freq_idx]
                    
                    st.metric("Amplitudenreserve", f"{gain_margin_exact:.1f} dB")
                    st.write(f"Bei ω = {phase_crossover_freq:.3f} rad/s")
                else:
                    st.metric("Amplitudenreserve", "∞ (kein -180° Durchgang)")
                    gain_margin_exact = float('inf')
            else:
                st.metric("Amplitudenreserve", "∞ (kein -180° Durchgang)")
                gain_margin_exact = float('inf')
            
            # Phasenreserve (exakt)
            zero_crossings = np.where(np.diff(np.signbit(mag_exact)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                if idx < len(omega_exact) - 1:
                    gain_crossover_freq = np.interp(0, [mag_exact[idx], mag_exact[idx+1]], 
                                                  [omega_exact[idx], omega_exact[idx+1]])
                    freq_idx = np.argmin(np.abs(omega_exact - gain_crossover_freq))
                    phase_at_crossover = phase_exact[freq_idx]
                    phase_margin_exact = 180 + phase_at_crossover
                    
                    st.metric("Phasenreserve", f"{phase_margin_exact:.1f}°")
                    st.write(f"Bei ω = {gain_crossover_freq:.3f} rad/s")
                else:
                    st.metric("Phasenreserve", "N/A")
                    phase_margin_exact = None
            else:
                st.metric("Phasenreserve", "N/A")
                phase_margin_exact = None
        
        with col2:
            st.markdown("##### 📐 Asymptotische Näherung")
            
            # Amplitudenreserve (asymptotisch)
            phase_shifted_asymp = phase_asymp + 180
            phase_crossings_asymp = np.where(np.diff(np.signbit(phase_shifted_asymp)))[0]
            
            if len(phase_crossings_asymp) > 0:
                idx = phase_crossings_asymp[0]
                if idx < len(omega_asymp) - 1:
                    phase_crossover_freq_asymp = np.interp(0, [phase_shifted_asymp[idx], phase_shifted_asymp[idx+1]], 
                                                         [omega_asymp[idx], omega_asymp[idx+1]])
                    freq_idx = np.argmin(np.abs(omega_asymp - phase_crossover_freq_asymp))
                    gain_margin_asymp = -mag_asymp[freq_idx]
                    
                    st.metric("Amplitudenreserve", f"{gain_margin_asymp:.1f} dB")
                    st.write(f"Bei ω = {phase_crossover_freq_asymp:.3f} rad/s")
                else:
                    st.metric("Amplitudenreserve", "∞ (kein -180° Durchgang)")
                    gain_margin_asymp = float('inf')
            else:
                st.metric("Amplitudenreserve", "∞ (kein -180° Durchgang)")
                gain_margin_asymp = float('inf')
            
            # Phasenreserve (asymptotisch)
            zero_crossings_asymp = np.where(np.diff(np.signbit(mag_asymp)))[0]
            if len(zero_crossings_asymp) > 0:
                idx = zero_crossings_asymp[0]
                if idx < len(omega_asymp) - 1:
                    gain_crossover_freq_asymp = np.interp(0, [mag_asymp[idx], mag_asymp[idx+1]], 
                                                        [omega_asymp[idx], omega_asymp[idx+1]])
                    freq_idx = np.argmin(np.abs(omega_asymp - gain_crossover_freq_asymp))
                    phase_at_crossover_asymp = phase_asymp[freq_idx]
                    phase_margin_asymp = 180 + phase_at_crossover_asymp
                    
                    st.metric("Phasenreserve", f"{phase_margin_asymp:.1f}°")
                    st.write(f"Bei ω = {gain_crossover_freq_asymp:.3f} rad/s")
                else:
                    st.metric("Phasenreserve", "N/A")
                    phase_margin_asymp = None
            else:
                st.metric("Phasenreserve", "N/A")
                phase_margin_asymp = None
        
        # Unterschiede hervorheben
        st.markdown("##### ⚖️ Vergleich der Methoden")
        
        diff_data = []
        
        # Amplitudenreserve-Unterschied
        if gain_margin_exact != float('inf') and gain_margin_asymp != float('inf'):
            gain_diff = abs(gain_margin_exact - gain_margin_asymp)
            diff_data.append({
                'Parameter': 'Amplitudenreserve',
                'Exakt': f"{gain_margin_exact:.1f} dB",
                'Asymptotisch': f"{gain_margin_asymp:.1f} dB",
                'Unterschied': f"{gain_diff:.1f} dB"
            })
        else:
            diff_data.append({
                'Parameter': 'Amplitudenreserve',
                'Exakt': "∞" if gain_margin_exact == float('inf') else f"{gain_margin_exact:.1f} dB",
                'Asymptotisch': "∞" if gain_margin_asymp == float('inf') else f"{gain_margin_asymp:.1f} dB",
                'Unterschied': "Methoden-abhängig"
            })
        
        # Phasenreserve-Unterschied
        if phase_margin_exact is not None and phase_margin_asymp is not None:
            phase_diff = abs(phase_margin_exact - phase_margin_asymp)
            diff_data.append({
                'Parameter': 'Phasenreserve',
                'Exakt': f"{phase_margin_exact:.1f}°",
                'Asymptotisch': f"{phase_margin_asymp:.1f}°",
                'Unterschied': f"{phase_diff:.1f}°"
            })
        else:
            diff_data.append({
                'Parameter': 'Phasenreserve',
                'Exakt': "N/A" if phase_margin_exact is None else f"{phase_margin_exact:.1f}°",
                'Asymptotisch': "N/A" if phase_margin_asymp is None else f"{phase_margin_asymp:.1f}°",
                'Unterschied': "Methoden-abhängig"
            })
        
        import pandas as pd
        df_comparison = pd.DataFrame(diff_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Interpretationshinweise
        st.markdown("##### 📋 Interpretationshinweise")
        st.info("""
        **Exakte Berechnung:** Berücksichtigt die tatsächlichen kontinuierlichen Kurvenverläufe.
        **Asymptotische Näherung:** Verwendet idealisierte Knick-Approximationen (typisch in klassischer Regelungstechnik-Lehre).
        
        **Unterschiede entstehen durch:**
        - Komplexe Pole (gedämpfte Schwingungen vs. ideale Knicke)
        - Pol-Nullstellen-Dipole (werden asymptotisch oft vernachlässigt)
        - Systeme mit Asymptoten statt echten Durchgängen bei 0dB/-180°
        """)
    
    def _create_bode_plot_detailed_old(self, omega_vals, magnitude_db, phase_deg):
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
        
        st.plotly_chart(fig, use_container_width=True, key="bode_simple_plot")
        
        # Wichtige Frequenzen markieren (ohne G_s für alte Methode)
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
        
        st.plotly_chart(fig, use_container_width=True, key="nyquist_plot")
        
        # Nyquist-Stabilitätsanalyse
        self._nyquist_stability_analysis(real_vals, imag_vals, omega_vals)
    
    def _mark_important_frequencies(self, omega_vals, magnitude_db, phase_deg, G_s=None, s=None):
        """Markiere wichtige Frequenzen im Bode-Diagramm"""
        st.markdown("#### 🎯 Charakteristische Frequenzen")
        
        # Extrahiere Knickfrequenzen falls G_s verfügbar
        break_frequencies = []
        if G_s is not None and s is not None:
            break_frequencies = self._extract_break_frequencies(G_s, s)
        
        col_freq1, col_freq2, col_freq3 = st.columns(3)
        
        with col_freq1:
            st.markdown("**Crossover-Frequenzen:**")
            
            # Finde 0 dB Durchgang
            zero_crossings = np.where(np.diff(np.signbit(magnitude_db)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                if idx < len(omega_vals) - 1:
                    gain_crossover = np.interp(0, [magnitude_db[idx], magnitude_db[idx+1]], 
                                             [omega_vals[idx], omega_vals[idx+1]])
                    st.metric("0 dB Durchgang", f"{gain_crossover:.3f} rad/s")
                    st.write(f"= {gain_crossover/(2*np.pi):.3f} Hz")
                else:
                    st.metric("0 dB Durchgang", "Nicht im Bereich")
            else:
                st.metric("0 dB Durchgang", "Nicht gefunden")
            
            # Finde -180° Durchgang
            phase_shifted = phase_deg + 180
            phase_crossings = np.where(np.diff(np.signbit(phase_shifted)))[0]
            if len(phase_crossings) > 0:
                idx = phase_crossings[0]
                if idx < len(omega_vals) - 1:
                    phase_crossover = np.interp(0, [phase_shifted[idx], phase_shifted[idx+1]], 
                                               [omega_vals[idx], omega_vals[idx+1]])
                    st.metric("-180° Durchgang", f"{phase_crossover:.3f} rad/s")
                    st.write(f"= {phase_crossover/(2*np.pi):.3f} Hz")
                else:
                    st.metric("-180° Durchgang", "Nicht im Bereich")
            else:
                st.metric("-180° Durchgang", "Nicht gefunden")
        
        with col_freq2:
            st.markdown("**Bandbreiten:**")
            
            # -3dB Bandbreite
            dc_gain_db = magnitude_db[0]
            cutoff_level = dc_gain_db - 3
            below_cutoff = magnitude_db < cutoff_level
            
            if np.any(below_cutoff):
                cutoff_idx = np.where(below_cutoff)[0][0]
                if cutoff_idx > 0:
                    bandwidth_3db = np.interp(cutoff_level, 
                                            [magnitude_db[cutoff_idx-1], magnitude_db[cutoff_idx]],
                                            [omega_vals[cutoff_idx-1], omega_vals[cutoff_idx]])
                    st.metric("-3dB Bandbreite", f"{bandwidth_3db:.3f} rad/s")
                    st.write(f"= {bandwidth_3db/(2*np.pi):.3f} Hz")
                else:
                    st.metric("-3dB Bandbreite", "> Max ω")
            else:
                st.metric("-3dB Bandbreite", "> Max ω")
            
            # Resonanzfrequenz (Peak)
            max_gain_idx = np.argmax(magnitude_db)
            resonance_freq = omega_vals[max_gain_idx]
            resonance_gain = magnitude_db[max_gain_idx]
            
            st.metric("Resonanzfrequenz", f"{resonance_freq:.3f} rad/s")
            st.write(f"Peak: {resonance_gain:.1f} dB")
        
        with col_freq3:
            st.markdown("**Knickfrequenzen:**")
            
            if break_frequencies:
                # Zeige die ersten Knickfrequenzen
                for i, (freq, btype) in enumerate(break_frequencies[:4]):
                    type_symbol = "🔴" if "pole" in btype else "🔵"
                    type_name = "Pol" if "pole" in btype else "Nullst."
                    if "complex" in btype:
                        type_name += " (kx)"
                    
                    st.write(f"{type_symbol} {freq:.2f} rad/s ({type_name})")
                
                if len(break_frequencies) > 4:
                    st.write(f"... und {len(break_frequencies)-4} weitere")
            else:
                st.write("Keine Knickfrequenzen identifiziert")
        
        # Zusätzliche Knickfrequenz-Tabelle
        if break_frequencies:
            st.markdown("#### 📊 Knickfrequenz-Übersicht")
            
            break_info = []
            for freq, btype in break_frequencies:
                if "pole" in btype:
                    symbol = "🔴"
                    desc = "Pol"
                    effect_mag = "-20 dB/Dek"
                    effect_phase = "-90°"
                else:
                    symbol = "🔵" 
                    desc = "Nullstelle"
                    effect_mag = "+20 dB/Dek"
                    effect_phase = "+90°"
                
                if "complex" in btype:
                    desc += " (komplex)"
                    effect_mag = effect_mag.replace("20", "40")
                    effect_phase = effect_phase.replace("90", "180")
                
                break_info.append({
                    'Typ': f"{symbol} {desc}",
                    'Frequenz [rad/s]': f"{freq:.3f}",
                    'Frequenz [Hz]': f"{freq/(2*np.pi):.3f}",
                    'Magnitude-Effekt': effect_mag,
                    'Phasen-Effekt': effect_phase
                })
            
    def _extract_break_frequencies(self, G_s, s):
        """Extrahiere Knickfrequenzen aus der Übertragungsfunktion"""
        try:
            # Vereinfache die Übertragungsfunktion
            G_s_simplified = sp.simplify(G_s)
            
            # Extrahiere Zähler und Nenner
            num, den = sp.fraction(G_s_simplified)
            
            # Finde Nullstellen (Zähler)
            zeros = sp.solve(num, s)
            
            # Finde Pole (Nenner)
            poles = sp.solve(den, s)
            
            break_frequencies = []
            
            # Verarbeite Pole
            for pole in poles:
                if pole.is_real and pole.is_negative:
                    # Reeller Pol: ω_b = |pole|
                    break_freq = float(abs(pole))
                    break_frequencies.append((break_freq, "pole_real"))
                elif pole.is_real and pole == 0:
                    # Pol bei s=0 (Integrator)
                    continue
                elif not pole.is_real:
                    # Komplexer Pol: ω_b = |pole|
                    try:
                        break_freq = float(abs(pole))
                        break_frequencies.append((break_freq, "pole_complex"))
                    except:
                        continue
            
            # Verarbeite Nullstellen
            for zero in zeros:
                if zero.is_real and zero.is_negative:
                    # Reelle Nullstelle: ω_b = |zero|
                    break_freq = float(abs(zero))
                    break_frequencies.append((break_freq, "zero_real"))
                elif zero.is_real and zero == 0:
                    # Nullstelle bei s=0 (Differenzierer)
                    continue
                elif not zero.is_real:
                    # Komplexe Nullstelle: ω_b = |zero|
                    try:
                        break_freq = float(abs(zero))
                        break_frequencies.append((break_freq, "zero_complex"))
                    except:
                        continue
            
            # Sortiere nach Frequenz
            break_frequencies.sort(key=lambda x: x[0])
            
            return break_frequencies
            
        except Exception as e:
            st.warning(f"Knickfrequenz-Extraktion fehlgeschlagen: {e}")
            return []
    
    def _add_break_frequency_markers(self, fig, G_s, s, omega_vals, magnitude_db, phase_deg):
        """Füge Knickfrequenz-Marker zu den Bode-Diagrammen hinzu"""
        try:
            break_frequencies = self._extract_break_frequencies(G_s, s)
            
            if not break_frequencies:
                return
            
            for freq, btype in break_frequencies:
                # Prüfe ob Frequenz im dargestellten Bereich liegt
                if freq < omega_vals[0] or freq > omega_vals[-1]:
                    continue
                
                # Interpoliere Magnitude und Phase bei Knickfrequenz
                freq_idx = np.argmin(np.abs(omega_vals - freq))
                mag_at_break = magnitude_db[freq_idx]
                phase_at_break = phase_deg[freq_idx]
                
                # Marker-Eigenschaften basierend auf Typ
                if "pole" in btype:
                    color = 'red'
                    symbol = 'circle'
                    name_prefix = "Pol"
                else:
                    color = 'blue' 
                    symbol = 'square'
                    name_prefix = "Nullstelle"
                
                if "complex" in btype:
                    name_suffix = " (komplex)"
                    size = 12
                else:
                    name_suffix = ""
                    size = 10
                
                marker_name = f"{name_prefix}{name_suffix} @ {freq:.2f} rad/s"
                
                # Magnitude-Marker
                fig.add_trace(
                    go.Scatter(
                        x=[freq], 
                        y=[mag_at_break],
                        mode='markers',
                        marker=dict(
                            symbol=symbol,
                            size=size,
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        name=marker_name,
                        showlegend=True,
                        hovertemplate=f'{marker_name}<br>ω: {freq:.3f} rad/s<br>|G|: {mag_at_break:.1f} dB<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Phasen-Marker
                fig.add_trace(
                    go.Scatter(
                        x=[freq], 
                        y=[phase_at_break],
                        mode='markers',
                        marker=dict(
                            symbol=symbol,
                            size=size,
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        showlegend=False,  # Nicht doppelt in Legende
                        hovertemplate=f'{marker_name}<br>ω: {freq:.3f} rad/s<br>∠G: {phase_at_break:.1f}°<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Vertikale Hilfslinie (optional)
                fig.add_vline(
                    x=freq,
                    line_dash="dot",
                    line_color=color,
                    opacity=0.3,
                    annotation_text=f"ω_{name_prefix.lower()}={freq:.1f}",
                    annotation_position="top"
                )
        
        except Exception as e:
            st.warning(f"Knickfrequenz-Marker konnten nicht hinzugefügt werden: {e}")
    
    def _create_separate_magnitude_phase_plots(self, omega_vals, magnitude, phase_deg):
        
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
            st.plotly_chart(fig_mag, use_container_width=True, key="bode_magnitude_plot")
        
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
            st.plotly_chart(fig_phase, use_container_width=True, key="bode_phase_plot")
    
    def _analyze_stability_margins_detailed(self, omega_vals, magnitude_db, phase_deg, G_s=None, s=None):
        """Detaillierte Stabilitätsreserven-Analyse mit Knickfrequenz-Support"""
        
        col_margin1, col_margin2 = st.columns(2)
        
        with col_margin1:
            st.markdown("#### 📊 Amplitudenreserve")
            
            # Debug: Zeige Phase-Werte
            min_phase = np.min(phase_deg)
            max_phase = np.max(phase_deg)
            st.write(f"Debug: Phase-Bereich: {min_phase:.1f}° bis {max_phase:.1f}°")
            
            # Erweiterte Suche nach -180° Durchgang
            target_phase = -180.0
            phase_diff = np.abs(phase_deg - target_phase)
            min_diff_idx = np.argmin(phase_diff)
            min_phase_diff = phase_diff[min_diff_idx]
            
            st.write(f"Debug: Nähester Punkt zu -180°: {phase_deg[min_diff_idx]:.1f}° (Abweichung: {min_phase_diff:.1f}°)")
            
            # Verwende relaxierte Toleranz für -180° Durchgang
            if min_phase_diff < 10.0:  # ±10° Toleranz
                phase_crossover_freq = omega_vals[min_diff_idx]
                gain_margin = -magnitude_db[min_diff_idx]
                
                st.metric("Amplitudenreserve", f"{gain_margin:.1f} dB")
                st.write(f"Bei ω = {phase_crossover_freq:.3f} rad/s")
                st.write(f"Phase: {phase_deg[min_diff_idx]:.1f}° (Ziel: -180°)")
                
                if gain_margin > 6:
                    st.success("✅ Gut (> 6 dB)")
                elif gain_margin > 3:
                    st.warning("⚠️ Akzeptabel (3-6 dB)")
                else:
                    st.error("❌ Kritisch (< 3 dB)")
            else:
                # Suche nach tatsächlichen Nulldurchgängen (alte Methode als Fallback)
                phase_shifted = phase_deg + 180
                phase_crossings = np.where(np.diff(np.signbit(phase_shifted)))[0]
                
                if len(phase_crossings) > 0:
                    idx = phase_crossings[0]
                    if idx < len(omega_vals) - 1:
                        phase_crossover_freq = np.interp(0, [phase_shifted[idx], phase_shifted[idx+1]], 
                                                       [omega_vals[idx], omega_vals[idx+1]])
                        # Magnitude bei dieser Frequenz interpolieren
                        gain_at_crossover = np.interp(phase_crossover_freq, omega_vals, magnitude_db)
                        gain_margin = -gain_at_crossover
                        
                        st.metric("Amplitudenreserve", f"{gain_margin:.1f} dB")
                        st.write(f"Bei ω = {phase_crossover_freq:.3f} rad/s")
                        
                        if gain_margin > 6:
                            st.success("✅ Gut (> 6 dB)")
                        elif gain_margin > 3:
                            st.warning("⚠️ Akzeptabel (3-6 dB)")
                        else:
                            st.error("❌ Kritisch (< 3 dB)")
                    else:
                        st.write("Phasendurchgang außerhalb des Frequenzbereichs")
                else:
                    st.write("Kein -180° Durchgang gefunden (auch mit erweiterter Suche)")
        
        with col_margin2:
            st.markdown("#### 📐 Phasenreserve")
            
            # Finde 0 dB Durchgang (mit Toleranz)
            tolerance_db = 1.0  # ±1 dB Toleranz
            near_zero_db = np.abs(magnitude_db) < tolerance_db
            
            if np.any(near_zero_db):
                # Finde genaueste Annäherung an 0 dB
                closest_idx = np.argmin(np.abs(magnitude_db))
                gain_crossover_freq = omega_vals[closest_idx]
                phase_at_crossover = phase_deg[closest_idx]
                phase_margin = 180 + phase_at_crossover
                
                st.metric("Phasenreserve", f"{phase_margin:.1f}°")
                st.write(f"Bei ω = {gain_crossover_freq:.3f} rad/s")
                st.write(f"Magnitude: {magnitude_db[closest_idx]:.1f} dB (Ziel: 0 dB)")
                
                if phase_margin > 45:
                    st.success("✅ Gut (> 45°)")
                elif phase_margin > 30:
                    st.warning("⚠️ Akzeptabel (30-45°)")
                else:
                    st.error("❌ Kritisch (< 30°)")
            else:
                # Suche nach tatsächlichen Nulldurchgängen
                zero_crossings = np.where(np.diff(np.signbit(magnitude_db)))[0]
                if len(zero_crossings) > 0:
                    idx = zero_crossings[0]
                    if idx < len(omega_vals) - 1:
                        gain_crossover_freq = np.interp(0, [magnitude_db[idx], magnitude_db[idx+1]], 
                                                      [omega_vals[idx], omega_vals[idx+1]])
                        # Phase bei dieser Frequenz interpolieren
                        phase_at_crossover = np.interp(gain_crossover_freq, omega_vals, phase_deg)
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
                        st.write("0 dB Durchgang außerhalb des Frequenzbereichs")
                else:
                    # Knickfrequenz-basierter Fallback wenn kein 0 dB Durchgang
                    if G_s is not None and s is not None:
                        st.info("ℹ️ Klassische Knickfrequenz-Analyse")
                        try:
                            # Schätze Phasenreserve basierend auf Systemstruktur
                            phase_estimate = self._estimate_phase_margin_from_structure(G_s, s)
                            if phase_estimate is not None:
                                st.write(f"Geschätzte Phasenreserve: ~{phase_estimate:.1f}°")
                                st.caption("Basierend auf Pol-/Nullstellenstruktur")
                            else:
                                st.write("Struktur-basierte Schätzung nicht möglich")
                        except Exception as e:
                            st.write(f"Strukturanalyse nicht möglich: {e}")
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
    def _create_bode_plot(self, G_s, s, omega, show_asymptotic=False):
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
                self._create_bode_plot_detailed(omega_clean, magnitude_db, phase_deg, G_s, s, show_asymptotic)
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
        
        st.plotly_chart(fig, use_container_width=True, key="nyquist_extended_plot")
        
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
            
            st.plotly_chart(fig, use_container_width=True, key="root_locus_plot")
            
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
    
    def comparison_analysis(self):
        """Vergleichsanalyse mehrerer Übertragungsfunktionen"""
        st.subheader("⚖️ Vergleichsanalyse")
        
        st.markdown("""
        **Vergleichsanalyse:** Analysiere und vergleiche mehrere Übertragungsfunktionen gleichzeitig.
        Perfekt für Systemvergleiche, Reglerentwurf und Parameteroptimierung.
        """)
        
        # Anzahl der Systeme
        num_systems = st.number_input(
            "Anzahl der Systeme:", 
            min_value=2, 
            max_value=6, 
            value=3,
            key='num_systems_comp'
        )
        
        st.markdown("### 🎛️ Systemdefinitionen")
        
        systems = []
        labels = []
        
        # Systemeingaben
        for i in range(num_systems):
            st.markdown(f"#### System {i+1}")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                tf = st.text_input(
                    f"G{i+1}(s):",
                    value=f"1/(s+{i+1})" if i < 3 else "10/(s**2+2*s+10)",
                    key=f"system_comp_{i}",
                    help="Beispiele: 1/(s+1), 10/(s**2+2*s+10), (s+1)/(s**2+3*s+2)"
                )
                systems.append(tf)
            
            with col2:
                label = st.text_input(
                    f"Label:",
                    value=f"System {i+1}",
                    key=f"label_comp_{i}"
                )
                labels.append(label)
        
        # Vergleichsoptionen
        st.markdown("### ⚙️ Vergleichsoptionen")
        
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            st.markdown("**Darstellung:**")
            show_combined = st.checkbox(
                "Gesamtübertragungsfunktion anzeigen", 
                help="Zeigt das Produkt aller Systeme: G_gesamt = G1 × G2 × ...",
                key='show_combined_comp'
            )
            show_individual_margins = st.checkbox(
                "Einzelsystem-Reserven", 
                value=True,
                key='show_individual_margins_comp'
            )
        
        with col_opt2:
            st.markdown("**Analyse:**")
            show_stability_margins = st.checkbox(
                "Stabilitätsreserven anzeigen", 
                value=True,
                help="Berechnet Amplituden- und Phasenreserven",
                key='show_stability_margins_comp'
            )
            show_characteristics = st.checkbox(
                "Systemcharakteristik", 
                value=True,
                key='show_characteristics_comp'
            )
        
        with col_opt3:
            st.markdown("**Plots:**")
            show_bode_comparison = st.checkbox(
                "Bode-Vergleich", 
                value=True,
                key='show_bode_comparison_comp'
            )
            show_nyquist_comparison = st.checkbox(
                "Nyquist-Vergleich", 
                value=True,
                key='show_nyquist_comparison_comp'
            )
        
        if st.button("🚀 Vergleichsanalyse starten", type="primary", key='start_comparison_analysis'):
            self._create_comparison_plots(
                systems, labels, show_combined, show_stability_margins,
                show_individual_margins, show_characteristics, 
                show_bode_comparison, show_nyquist_comparison
            )
    
    def _create_comparison_plots(self, systems, labels, show_combined=False, show_stability_margins=False,
                               show_individual_margins=True, show_characteristics=True,
                               show_bode_comparison=True, show_nyquist_comparison=True):
        """Erstelle Vergleichsdiagramme für mehrere Systeme"""
        
        st.markdown("### 📊 Vergleichsdiagramme")
        
        # Plots vorbereiten
        if show_nyquist_comparison:
            fig_nyquist = go.Figure()
        
        if show_bode_comparison:
            fig_bode = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Amplitudengang Vergleich', 'Phasengang Vergleich'),
                vertical_spacing=0.1
            )
        
        # Daten für Gesamtübertragungsfunktion sammeln
        valid_systems = []
        valid_labels = []
        system_data = []
        
        # Einzelsysteme verarbeiten
        for i, (tf_str, label) in enumerate(zip(systems, labels)):
            try:
                # Parse System
                s, omega = sp.symbols('s omega', real=True)
                symbols_dict = {'s': s, 'omega': omega, 'j': sp.I}
                
                G_s = safe_sympify(tf_str, symbols_dict)
                
                # Konvertiere zu SymPy Ausdruck falls nötig
                if not hasattr(G_s, 'subs'):
                    G_s = sp.sympify(G_s)
                
                valid_systems.append(G_s)
                valid_labels.append(label)
                
                # Frequenzgang berechnen
                omega_vals = np.logspace(-2, 3, 1000)
                j = sp.I
                G_jw = G_s.subs(s, j*omega)
                
                # Numerische Auswertung
                if G_jw.is_number:
                    G_vals = np.full_like(omega_vals, complex(G_jw), dtype=complex)
                else:
                    G_func = sp.lambdify(omega, G_jw, 'numpy')
                    G_vals = G_func(omega_vals)
                
                # Filter finite values
                finite_mask = np.isfinite(G_vals)
                omega_clean = omega_vals[finite_mask]
                G_clean = G_vals[finite_mask]
                
                if len(G_clean) > 0:
                    real_vals = np.real(G_clean)
                    imag_vals = np.imag(G_clean)
                    magnitude_db = 20 * np.log10(np.abs(G_clean))
                    phase_deg = np.angle(G_clean) * 180 / np.pi
                    
                    # Speichere Systemdaten
                    system_data.append({
                        'label': label,
                        'G_s': G_s,
                        'omega_clean': omega_clean,
                        'magnitude_db': magnitude_db,
                        'phase_deg': phase_deg,
                        'real_vals': real_vals,
                        'imag_vals': imag_vals
                    })
                    
                    # Farbe für dieses System
                    color = f'hsl({i*40}, 70%, 50%)'
                    
                    # Nyquist-Plot
                    if show_nyquist_comparison:
                        fig_nyquist.add_trace(go.Scatter(
                            x=real_vals, y=imag_vals,
                            mode='lines',
                            name=label,
                            line=dict(color=color, width=2),
                            hovertemplate=f'{label}<br>Real: %{{x:.3f}}<br>Imag: %{{y:.3f}}<extra></extra>'
                        ))
                    
                    # Bode-Plot
                    if show_bode_comparison:
                        fig_bode.add_trace(
                            go.Scatter(
                                x=omega_clean, y=magnitude_db,
                                mode='lines',
                                name=label,
                                line=dict(color=color, width=2),
                                hovertemplate=f'{label}<br>ω: %{{x:.3f}}<br>|G|: %{{y:.1f}} dB<extra></extra>'
                            ),
                            row=1, col=1
                        )
                        
                        fig_bode.add_trace(
                            go.Scatter(
                                x=omega_clean, y=phase_deg,
                                mode='lines',
                                name=label,
                                line=dict(color=color, width=2),
                                showlegend=False,
                                hovertemplate=f'{label}<br>ω: %{{x:.3f}}<br>∠G: %{{y:.1f}}°<extra></extra>'
                            ),
                            row=2, col=1
                        )
            
            except Exception as e:
                st.error(f"Fehler bei System {i+1} ({label}): {e}")
                continue
        
        # Gesamtübertragungsfunktion hinzufügen
        if show_combined and len(valid_systems) > 1:
            self._add_combined_system_to_plots(
                valid_systems, fig_nyquist if show_nyquist_comparison else None, 
                fig_bode if show_bode_comparison else None, s, omega
            )
        
        # Plots konfigurieren und anzeigen
        if show_nyquist_comparison:
            self._configure_and_show_nyquist_plot(fig_nyquist)
        
        if show_bode_comparison:
            # Füge Knickfrequenz-Marker für alle Systeme hinzu
            for data in system_data:
                self._add_break_frequency_markers(
                    fig_bode, data['G_s'], s, data['omega_clean'], 
                    data['magnitude_db'], data['phase_deg']
                )
            
            self._configure_and_show_bode_plot(fig_bode)
        
        # Stabilitätsreserven für Gesamtfunktion
        if show_combined and show_stability_margins and len(valid_systems) > 1:
            self._analyze_combined_stability_margins(valid_systems, s, omega)
        
        # Einzelsystem-Analysen
        if show_individual_margins or show_characteristics:
            self._show_individual_system_analysis(system_data, show_individual_margins, show_characteristics)
    
    def _add_combined_system_to_plots(self, valid_systems, fig_nyquist, fig_bode, s, omega):
        """Füge Gesamtübertragungsfunktion zu den Plots hinzu"""
        try:
            # Berechne Gesamtübertragungsfunktion G_gesamt = G1 * G2 * G3 * ...
            G_gesamt = valid_systems[0]
            for system in valid_systems[1:]:
                G_gesamt = G_gesamt * system
            
            # Frequenzgang der Gesamtfunktion berechnen
            omega_vals = np.logspace(-2, 3, 1000)
            j = sp.I
            G_gesamt_jw = G_gesamt.subs(s, j*omega)
            
            if G_gesamt_jw.is_number:
                G_gesamt_vals = np.full_like(omega_vals, complex(G_gesamt_jw), dtype=complex)
            else:
                G_gesamt_func = sp.lambdify(omega, G_gesamt_jw, 'numpy')
                G_gesamt_vals = G_gesamt_func(omega_vals)
            
            # Filter finite values
            finite_mask = np.isfinite(G_gesamt_vals)
            omega_clean = omega_vals[finite_mask]
            G_clean = G_gesamt_vals[finite_mask]
            
            if len(G_clean) > 0:
                real_vals = np.real(G_clean)
                imag_vals = np.imag(G_clean)
                magnitude_db = 20 * np.log10(np.abs(G_clean))
                phase_deg = np.angle(G_clean) * 180 / np.pi
                
                # Speichere für Stabilitätsanalyse
                self._combined_system_data = {
                    'omega_clean': omega_clean,
                    'magnitude_db': magnitude_db,
                    'phase_deg': phase_deg,
                    'G_gesamt': G_gesamt
                }
                
                # Zu Nyquist-Plot hinzufügen
                if fig_nyquist is not None:
                    fig_nyquist.add_trace(go.Scatter(
                        x=real_vals, y=imag_vals,
                        mode='lines',
                        name='Gesamtfunktion G₁×G₂×...',
                        line=dict(color='black', width=3, dash='dash'),
                        hovertemplate='Gesamt<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>'
                    ))
                
                # Zu Bode-Plot hinzufügen
                if fig_bode is not None:
                    fig_bode.add_trace(
                        go.Scatter(
                            x=omega_clean, y=magnitude_db,
                            mode='lines',
                            name='Gesamtfunktion',
                            line=dict(color='black', width=3, dash='dash'),
                            hovertemplate='ω: %{x:.3f}<br>|G|: %{y:.1f} dB<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    fig_bode.add_trace(
                        go.Scatter(
                            x=omega_clean, y=phase_deg,
                            mode='lines',
                            name='Gesamtfunktion',
                            line=dict(color='black', width=3, dash='dash'),
                            showlegend=False,
                            hovertemplate='ω: %{x:.3f}<br>∠G: %{y:.1f}°<extra></extra>'
                        ),
                        row=2, col=1
                    )
            
        except Exception as e:
            st.error(f"❌ Fehler bei Gesamtfunktions-Berechnung: {e}")
    
    def _configure_and_show_nyquist_plot(self, fig_nyquist):
        """Konfiguriere und zeige Nyquist-Plot"""
        # Kritischer Punkt hinzufügen
        fig_nyquist.add_trace(go.Scatter(
            x=[-1], y=[0],
            mode='markers',
            marker=dict(symbol='x', size=15, color='red', line=dict(width=3)),
            name='Kritischer Punkt (-1+0j)',
            hovertemplate='Kritischer Punkt<br>Real: -1<br>Imag: 0<extra></extra>'
        ))
        
        fig_nyquist.update_layout(
            title="Nyquist-Diagramm Vergleich",
            xaxis_title="Realteil",
            yaxis_title="Imaginärteil",
            showlegend=True,
            height=500,
            xaxis=dict(showgrid=True, scaleanchor="y", scaleratio=1),
            yaxis=dict(showgrid=True)
        )
        
        st.plotly_chart(fig_nyquist, use_container_width=True, key="comparison_nyquist_plot")
    
    def _configure_and_show_bode_plot(self, fig_bode):
        """Konfiguriere und zeige Bode-Plot"""
        # Layout
        fig_bode.update_xaxes(type="log", title_text="Frequenz ω [rad/s]", row=2, col=1)
        fig_bode.update_xaxes(type="log", row=1, col=1)
        fig_bode.update_yaxes(title_text="Magnitude [dB]", row=1, col=1)
        fig_bode.update_yaxes(title_text="Phase [°]", row=2, col=1)
        
        # Referenzlinien
        fig_bode.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig_bode.add_hline(y=-180, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig_bode.update_layout(
            title="Bode-Diagramm Vergleich",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig_bode, use_container_width=True, key="comparison_bode_plot")
    
    def _analyze_combined_stability_margins(self, valid_systems, s, omega):
        """Analysiere Stabilitätsreserven der Gesamtfunktion"""
        if not hasattr(self, '_combined_system_data'):
            st.error("❌ Gesamtfunktionsdaten nicht verfügbar")
            return
        
        data = self._combined_system_data
        omega_clean = data['omega_clean']
        magnitude_db = data['magnitude_db']
        phase_deg = data['phase_deg']
        G_gesamt = data['G_gesamt']
        
        st.markdown("---")
        st.markdown("### 📏 Stabilitätsreserven der Gesamtübertragungsfunktion")
        
        # Extrahiere Knickfrequenzen für präzise Analyse
        break_frequencies = self._extract_break_frequencies(G_gesamt, s)
        
        col_margin1, col_margin2 = st.columns(2)
        
        with col_margin1:
            st.markdown("#### 📊 Amplitudenreserve (Gesamtfunktion)")
            
            # Erweiterte -180° Durchgang-Suche
            phase_shifted = phase_deg + 180
            phase_crossings = np.where(np.diff(np.signbit(phase_shifted)))[0]
            
            gain_margin = None
            phase_crossover_freq = None
            
            if len(phase_crossings) > 0:
                # Exakter Durchgang gefunden
                idx = phase_crossings[0]
                if idx < len(omega_clean) - 1:
                    phase_crossover_freq = np.interp(0, [phase_shifted[idx], phase_shifted[idx+1]], 
                                                   [omega_clean[idx], omega_clean[idx+1]])
                    freq_idx = np.argmin(np.abs(omega_clean - phase_crossover_freq))
                    gain_margin = -magnitude_db[freq_idx]
                    
                    st.metric("Amplitudenreserve", f"{gain_margin:.1f} dB")
                    st.write(f"Bei ω = {phase_crossover_freq:.3f} rad/s")
                    
                    # Bewertung mit Icon
                    if gain_margin > 12:
                        st.success("✅ Sehr gut (> 12 dB)")
                    elif gain_margin > 6:
                        st.success("✅ Gut (6-12 dB)")
                    elif gain_margin > 3:
                        st.warning("⚠️ Akzeptabel (3-6 dB)")
                    else:
                        st.error("❌ Kritisch (< 3 dB)")
            else:
                # Kein exakter Durchgang - analysiere basierend auf Knickfrequenzen
                st.markdown("**Knickfrequenz-basierte Analyse:**")
                
                # Berechne asymptotische Phase basierend auf Systemstruktur
                asymptotic_phase_at_high_freq = self._calculate_asymptotic_phase_from_breaks(G_gesamt, s, break_frequencies)
                
                st.write(f"Asymptotische Phase (ω→∞): {asymptotic_phase_at_high_freq:.1f}°")
                
                if asymptotic_phase_at_high_freq <= -180:
                    # System erreicht theoretisch -180°
                    # Berechne Frequenz basierend auf Knickfrequenzen
                    estimated_phase_crossing = self._estimate_phase_crossing_from_breaks(
                        G_gesamt, s, break_frequencies, -180
                    )
                    
                    if estimated_phase_crossing:
                        # Berechne Magnitude bei dieser Frequenz
                        estimated_mag = self._calculate_magnitude_at_frequency(
                            G_gesamt, s, omega, estimated_phase_crossing
                        )
                        estimated_gain_margin = -estimated_mag
                        
                        st.metric("Amplitudenreserve (berechnet)", f"{estimated_gain_margin:.1f} dB")
                        st.write(f"Bei ω ≈ {estimated_phase_crossing:.3f} rad/s")
                        st.info("ℹ️ Basiert auf Knickfrequenz-Analyse")
                        
                        # Bewertung
                        if estimated_gain_margin > 12:
                            st.success("✅ Sehr gut (> 12 dB)")
                        elif estimated_gain_margin > 6:
                            st.success("✅ Gut (6-12 dB)")
                        elif estimated_gain_margin > 3:
                            st.warning("⚠️ Akzeptabel (3-6 dB)")
                        else:
                            st.error("❌ Kritisch (< 3 dB)")
                    else:
                        st.metric("Amplitudenreserve", "∞ (praktisch stabil)")
                        st.success("✅ System erreicht -180° nur bei ω→∞")
                else:
                    # System erreicht nie -180°
                    st.metric("Amplitudenreserve", "∞ (unendlich)")
                    st.success("✅ System erreicht nie -180°")
        
        with col_margin2:
            st.markdown("#### 📐 Phasenreserve (Gesamtfunktion)")
            
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
                    
                    # Bewertung mit Icon
                    if phase_margin > 60:
                        st.success("✅ Sehr gut (> 60°)")
                    elif phase_margin > 45:
                        st.success("✅ Gut (45-60°)")
                    elif phase_margin > 30:
                        st.warning("⚠️ Akzeptabel (30-45°)")
                    else:
                        st.error("❌ Kritisch (< 30°)")
                else:
                    st.metric("Phasenreserve", "N/A")
            else:
                # Kein 0 dB Durchgang - analysiere basierend auf Knickfrequenzen
                st.markdown("**0 dB Knickfrequenz-Analyse:**")
                
                # Schätze 0 dB Durchgang basierend auf DC-Verstärkung und Systemordnung
                estimated_gain_crossing = self._estimate_gain_crossing_from_breaks(
                    G_gesamt, s, break_frequencies, 0
                )
                
                if estimated_gain_crossing:
                    estimated_phase = self._calculate_phase_at_frequency(
                        G_gesamt, s, omega, estimated_gain_crossing
                    )
                    estimated_phase_margin = 180 + estimated_phase
                    
                    st.metric("Phasenreserve (berechnet)", f"{estimated_phase_margin:.1f}°")
                    st.write(f"Bei ω ≈ {estimated_gain_crossing:.3f} rad/s")
                    st.info("ℹ️ Basiert auf Knickfrequenz-Analyse")
                    
                    # Bewertung
                    if estimated_phase_margin > 60:
                        st.success("✅ Sehr gut (> 60°)")
                    elif estimated_phase_margin > 45:
                        st.success("✅ Gut (45-60°)")
                    elif estimated_phase_margin > 30:
                        st.warning("⚠️ Akzeptabel (30-45°)")
                    else:
                        st.error("❌ Kritisch (< 30°)")
                else:
                    # Analysiere warum kein Durchgang
                    dc_gain = self._get_dc_gain(G_gesamt, s)
                    if dc_gain > 1:  # > 0 dB
                        st.metric("Phasenreserve", "System hat hohe DC-Verstärkung")
                        st.info("ℹ️ 0 dB Durchgang bei sehr hohen Frequenzen")
                    else:
                        st.metric("Phasenreserve", "N/A (Mag. unter 0 dB)")
                        st.warning("⚠️ Magnitude bleibt unter 0 dB")
        
        # Zusätzliche Gesamtfunktions-Eigenschaften
        st.markdown("#### 🎯 Gesamtfunktions-Charakteristik")
        
        col_char1, col_char2, col_char3 = st.columns(3)
        
        with col_char1:
            # DC-Verstärkung
            try:
                dc_gain_db = magnitude_db[0]  # Bei niedrigster Frequenz
                st.metric("DC-Verstärkung", f"{dc_gain_db:.1f} dB")
                
                # Linear auch anzeigen
                dc_gain_linear = 10**(dc_gain_db/20)
                st.write(f"Linear: {dc_gain_linear:.3f}")
            except:
                st.metric("DC-Verstärkung", "N/A")
        
        with col_char2:
            # Bandbreite (-3dB Punkt)
            try:
                dc_gain_db = magnitude_db[0]
                cutoff_level = dc_gain_db - 3
                below_cutoff = magnitude_db < cutoff_level
                if np.any(below_cutoff):
                    cutoff_idx = np.where(below_cutoff)[0][0]
                    if cutoff_idx > 0:
                        bandwidth = np.interp(cutoff_level, 
                                            [magnitude_db[cutoff_idx-1], magnitude_db[cutoff_idx]],
                                            [omega_clean[cutoff_idx-1], omega_clean[cutoff_idx]])
                        st.metric("Bandbreite (-3dB)", f"{bandwidth:.2f} rad/s")
                        st.write(f"= {bandwidth/(2*np.pi):.2f} Hz")
                    else:
                        st.metric("Bandbreite (-3dB)", "> Max ω")
                else:
                    st.metric("Bandbreite (-3dB)", "> Max ω")
            except:
                st.metric("Bandbreite (-3dB)", "N/A")
        
        with col_char3:
            # Knickfrequenzen anzeigen
            st.markdown("**Knickfrequenzen:**")
            if break_frequencies:
                for i, (freq, btype) in enumerate(break_frequencies[:3]):  # Zeige nur erste 3
                    type_symbol = "×" if "pole" in btype else "○"
                    st.write(f"{type_symbol} {freq:.2f} rad/s")
                if len(break_frequencies) > 3:
                    st.write(f"... und {len(break_frequencies)-3} weitere")
            else:
                st.write("Keine Knickfrequenzen")
        
        # Zeige Knickfrequenzen als Info
        if break_frequencies:
            st.markdown("#### 📍 Identifizierte Knickfrequenzen")
            break_info = []
            for freq, btype in break_frequencies:
                if "pole" in btype:
                    symbol = "🔴"
                    desc = "Pol"
                else:
                    symbol = "🔵" 
                    desc = "Nullstelle"
                
                if "complex" in btype:
                    desc += " (komplex)"
                
                break_info.append({
                    'Typ': f"{symbol} {desc}",
                    'Frequenz [rad/s]': f"{freq:.3f}",
                    'Frequenz [Hz]': f"{freq/(2*np.pi):.3f}"
                })
            
            import pandas as pd
            df_breaks = pd.DataFrame(break_info)
            st.dataframe(df_breaks, use_container_width=True)
    
    def _extract_break_frequencies(self, G_s, s):
        """Extrahiere alle Knickfrequenzen aus der Übertragungsfunktion"""
        try:
            break_frequencies = []
            
            # Numerator und Denominator extrahieren
            num = sp.numer(G_s)
            den = sp.denom(G_s)
            
            # Pole verarbeiten
            poles = sp.solve(den, s)
            for pole in poles:
                pole_val = complex(pole.evalf())
                if abs(pole_val.imag) < 1e-10:  # Reeller Pol
                    if pole_val.real < 0:  # Stabiler Pol
                        break_freq = abs(pole_val.real)
                        if break_freq > 1e-6:  # Ignoriere Pole bei s≈0
                            break_frequencies.append((break_freq, 'pole'))
                else:  # Komplexer Pol
                    if pole_val.real < 0:  # Stabiler komplexer Pol
                        break_freq = abs(pole_val)  # Eigenfrequenz ωn
                        if break_freq > 1e-6:
                            break_frequencies.append((break_freq, 'complex_pole'))
            
            # Nullstellen verarbeiten
            zeros = sp.solve(num, s)
            for zero in zeros:
                zero_val = complex(zero.evalf())
                if abs(zero_val.imag) < 1e-10:  # Reelle Nullstelle
                    break_freq = abs(zero_val.real)
                    if break_freq > 1e-6:
                        break_frequencies.append((break_freq, 'zero'))
                else:  # Komplexe Nullstelle
                    break_freq = abs(zero_val)
                    if break_freq > 1e-6:
                        break_frequencies.append((break_freq, 'complex_zero'))
            
            # Sortiere nach Frequenz
            break_frequencies.sort(key=lambda x: x[0])
            
            return break_frequencies
            
        except Exception as e:
            st.warning(f"Knickfrequenz-Extraktion fehlgeschlagen: {e}")
            return []
    
    def _calculate_asymptotic_phase_from_breaks(self, G_s, s, break_frequencies):
        """Berechne asymptotische Phase basierend auf Knickfrequenzen"""
        try:
            # DC-Phase
            dc_gain = self._get_dc_gain(G_s, s)
            phase_offset = 180 if dc_gain < 0 else 0
            
            # Systemordnung bestimmen
            num = sp.numer(G_s)
            den = sp.denom(G_s)
            
            num_degree = sp.degree(num, s) if sp.degree(num, s) is not None else 0
            den_degree = sp.degree(den, s) if sp.degree(den, s) is not None else 0
            
            system_order = int(den_degree) - int(num_degree)
            
            # Asymptotische Phase = -90° × Systemordnung + DC-Offset
            asymptotic_phase = -90 * system_order + phase_offset
            
            # Berücksichtige Integrator-Terme (Pole bei s=0)
            poles = sp.solve(den, s)
            integrator_count = sum(1 for pole in poles if abs(complex(pole.evalf())) < 1e-6)
            asymptotic_phase -= 90 * integrator_count
            
            return asymptotic_phase
            
        except Exception:
            return -180  # Konservative Schätzung
    
    def _estimate_phase_crossing_from_breaks(self, G_s, s, break_frequencies, target_phase):
        """Schätze Frequenz für Phasendurchgang basierend auf Knickfrequenzen"""
        try:
            if not break_frequencies:
                return None
            
            # Für einfache Systeme: Grobe Schätzung basierend auf höchster Knickfrequenz
            # und Systemordnung
            highest_break = max(break_frequencies, key=lambda x: x[0])[0]
            
            # Systemordnung
            num = sp.numer(G_s)
            den = sp.denom(G_s)
            num_degree = sp.degree(num, s) if sp.degree(num, s) is not None else 0
            den_degree = sp.degree(den, s) if sp.degree(den, s) is not None else 0
            system_order = int(den_degree) - int(num_degree)
            
            # Grobe Schätzung: -180° wird etwa bei 10× der höchsten Knickfrequenz erreicht
            # für Systeme 2. Ordnung
            if system_order >= 2:
                estimated_freq = highest_break * 10
                return estimated_freq
            
            return None
            
        except Exception:
            return None
    
    def _estimate_gain_crossing_from_breaks(self, G_s, s, break_frequencies, target_gain_db):
        """Schätze Frequenz für Verstärkungsdurchgang basierend auf Knickfrequenzen"""
        try:
            # DC-Verstärkung
            dc_gain = self._get_dc_gain(G_s, s)
            dc_gain_db = 20 * np.log10(abs(dc_gain)) if dc_gain != 0 else -np.inf
            
            if dc_gain_db <= target_gain_db:
                return None  # 0 dB nie erreicht
            
            # Systemordnung
            num = sp.numer(G_s)
            den = sp.denom(G_s)
            num_degree = sp.degree(num, s) if sp.degree(num, s) is not None else 0
            den_degree = sp.degree(den, s) if sp.degree(den, s) is not None else 0
            system_order = int(den_degree) - int(num_degree)
            
            if not break_frequencies:
                return None
            
            # Einfache Schätzung basierend auf Systemordnung und DC-Verstärkung
            # Für System n-ter Ordnung: -20n dB/Dekade Abfall
            if system_order > 0:
                # Benötigte Dekaden für Abfall von DC-Gain auf target_gain
                decades_needed = (dc_gain_db - target_gain_db) / (20 * system_order)
                
                # Startfrequenz: Erste signifikante Knickfrequenz
                start_freq = min(break_frequencies, key=lambda x: x[0])[0]
                
                # Geschätzte Crossover-Frequenz
                estimated_freq = start_freq * (10 ** decades_needed)
                return estimated_freq
            
            return None
            
        except Exception:
            return None
    
    def _calculate_magnitude_at_frequency(self, G_s, s, omega, target_freq):
        """Berechne Magnitude bei einer bestimmten Frequenz"""
        try:
            j = sp.I
            G_jw = G_s.subs(s, j * target_freq)
            
            if G_jw.is_number:
                mag_val = abs(complex(G_jw))
            else:
                G_func = sp.lambdify(omega, G_jw, 'numpy')
                mag_val = abs(G_func(target_freq))
            
            mag_db = 20 * np.log10(mag_val) if mag_val > 0 else -np.inf
            return mag_db
            
        except Exception:
            return 0
    
    def _calculate_phase_at_frequency(self, G_s, s, omega, target_freq):
        """Berechne Phase bei einer bestimmten Frequenz"""
        try:
            j = sp.I
            G_jw = G_s.subs(s, j * target_freq)
            
            if G_jw.is_number:
                phase_val = np.angle(complex(G_jw))
            else:
                G_func = sp.lambdify(omega, G_jw, 'numpy')
                phase_val = np.angle(G_func(target_freq))
            
            phase_deg = phase_val * 180 / np.pi
            return phase_deg
            
        except Exception:
            return -180
    
    def _calculate_magnitude_at_frequency(self, G_s, s, omega, target_freq):
        """Berechne Magnitude bei einer bestimmten Frequenz"""
        try:
            j = sp.I
            G_jw = G_s.subs(s, j * target_freq)
            
            if G_jw.is_number:
                mag_val = abs(complex(G_jw.evalf()))
            else:
                G_func = sp.lambdify(omega, G_jw, 'numpy')
                mag_val = abs(G_func(target_freq))
            
            mag_db = 20 * np.log10(mag_val) if mag_val > 0 else -np.inf
            return mag_db
            
        except Exception:
            return 0
    
    def _calculate_phase_at_frequency(self, G_s, s, omega, target_freq):
        """Berechne Phase bei einer bestimmten Frequenz"""
        try:
            j = sp.I
            G_jw = G_s.subs(s, j * target_freq)
            
            if G_jw.is_number:
                phase_val = np.angle(complex(G_jw.evalf())) * 180 / np.pi
            else:
                G_func = sp.lambdify(omega, G_jw, 'numpy')
                phase_val = np.angle(G_func(target_freq)) * 180 / np.pi
            
            return phase_val
            
        except Exception:
            return 0
    
    def _get_dc_gain(self, G_s, s):
        """Ermittle DC-Verstärkung"""
        try:
            dc_gain = float(G_s.subs(s, 0))
            return dc_gain
        except:
            # Falls Pol bei s=0, berechne Grenzwert
            try:
                # Für Integrator-Systeme ist DC-Verstärkung unendlich
                limit_val = sp.limit(G_s, s, 0)
                if limit_val.is_infinite:
                    return float('inf')
                else:
                    return float(limit_val)
            except:
                return 1.0  # Fallback
    
    def _calculate_asymptotic_phase_limit(self, G_s, s):
        """Berechne asymptotische Phasengrenze für ω→∞"""
        try:
            # Systemordnung bestimmen
            num = sp.numer(G_s)
            den = sp.denom(G_s)
            
            num_degree = sp.degree(num, s) if sp.degree(num, s) is not None else 0
            den_degree = sp.degree(den, s) if sp.degree(den, s) is not None else 0
            
            system_order = int(den_degree) - int(num_degree)
            
            # Asymptotische Phase = -90° × Systemordnung
            asymptotic_phase = -90 * system_order
            
            # Berücksichtige DC-Verstärkung für Vorzeichen
            try:
                dc_gain = float(G_s.subs(s, 0))
                if dc_gain < 0:
                    asymptotic_phase += 180
            except:
                # Falls Pol bei s=0, analysiere Verhalten anders
                # Für Integrator-Terme
                poles = sp.solve(den, s)
                zeros_at_origin = sum(1 for pole in poles if abs(complex(pole.evalf())) < 1e-10)
                asymptotic_phase -= 90 * zeros_at_origin
            
            return asymptotic_phase
            
        except Exception:
            return -180  # Konservative Schätzung
    
    def _estimate_phase_crossing_frequency(self, omega_vals, phase_vals, target_phase):
        """Schätze Frequenz wo eine bestimmte Phase erreicht wird"""
        try:
            # Prüfe ob Trend zu target_phase führt
            phase_trend = phase_vals[-1] - phase_vals[0]
            freq_trend = omega_vals[-1] / omega_vals[0]
            
            if phase_trend == 0:
                return None
            
            # Lineare Extrapolation (logarithmisch in Frequenz)
            log_omega_vals = np.log10(omega_vals)
            
            # Finde Trend der letzten Punkte
            n_points = min(50, len(phase_vals) // 4)
            recent_log_omega = log_omega_vals[-n_points:]
            recent_phase = phase_vals[-n_points:]
            
            # Lineare Regression für Trend
            if len(recent_phase) > 1:
                slope = (recent_phase[-1] - recent_phase[0]) / (recent_log_omega[-1] - recent_log_omega[0])
                intercept = recent_phase[-1] - slope * recent_log_omega[-1]
                
                # Berechne log10(omega) für target_phase
                if abs(slope) > 1e-10:
                    log_omega_target = (target_phase - intercept) / slope
                    omega_target = 10 ** log_omega_target
                    
                    # Nur zurückgeben wenn sinnvoll (nicht zu weit extrapoliert)
                    if omega_target > omega_vals[-1] and omega_target < omega_vals[-1] * 1000:
                        return omega_target
            
            return None
            
        except Exception:
            return None
    
    def _estimate_magnitude_at_frequency(self, omega_vals, mag_vals, target_freq):
        """Schätze Magnitude bei einer bestimmten Frequenz"""
        try:
            if target_freq <= omega_vals[-1]:
                # Interpolation
                return np.interp(target_freq, omega_vals, mag_vals)
            else:
                # Extrapolation basierend auf Trend
                log_omega_vals = np.log10(omega_vals)
                
                # Verwende letzten Teil für Trend-Bestimmung
                n_points = min(50, len(mag_vals) // 4)
                recent_log_omega = log_omega_vals[-n_points:]
                recent_mag = mag_vals[-n_points:]
                
                # Lineare Regression für Trend
                if len(recent_mag) > 1:
                    slope = (recent_mag[-1] - recent_mag[0]) / (recent_log_omega[-1] - recent_log_omega[0])
                    intercept = recent_mag[-1] - slope * recent_log_omega[-1]
                    
                    log_target_freq = np.log10(target_freq)
                    estimated_mag = slope * log_target_freq + intercept
                    
                    return estimated_mag
                else:
                    return mag_vals[-1]
                    
        except Exception:
            return 0
    
    def _estimate_magnitude_crossing_frequency(self, omega_vals, mag_vals, target_mag):
        """Schätze Frequenz wo eine bestimmte Magnitude erreicht wird"""
        try:
            # Ähnlich zur Phase-Schätzung
            log_omega_vals = np.log10(omega_vals)
            
            # Finde Trend der letzten Punkte
            n_points = min(50, len(mag_vals) // 4)
            recent_log_omega = log_omega_vals[-n_points:]
            recent_mag = mag_vals[-n_points:]
            
            # Lineare Regression für Trend
            if len(recent_mag) > 1:
                slope = (recent_mag[-1] - recent_mag[0]) / (recent_log_omega[-1] - recent_log_omega[0])
                intercept = recent_mag[-1] - slope * recent_log_omega[-1]
                
                # Berechne log10(omega) für target_mag
                if abs(slope) > 1e-10:
                    log_omega_target = (target_mag - intercept) / slope
                    omega_target = 10 ** log_omega_target
                    
                    # Nur zurückgeben wenn sinnvoll
                    if omega_target > omega_vals[-1] and omega_target < omega_vals[-1] * 1000:
                        return omega_target
            
            return None
            
        except Exception:
            return None
    
    def _estimate_phase_at_frequency(self, omega_vals, phase_vals, target_freq):
        """Schätze Phase bei einer bestimmten Frequenz"""
        try:
            if target_freq <= omega_vals[-1]:
                # Interpolation
                return np.interp(target_freq, omega_vals, phase_vals)
            else:
                # Extrapolation basierend auf Trend
                log_omega_vals = np.log10(omega_vals)
                
                # Verwende letzten Teil für Trend-Bestimmung
                n_points = min(50, len(phase_vals) // 4)
                recent_log_omega = log_omega_vals[-n_points:]
                recent_phase = phase_vals[-n_points:]
                
                # Lineare Regression für Trend
                if len(recent_phase) > 1:
                    slope = (recent_phase[-1] - recent_phase[0]) / (recent_log_omega[-1] - recent_log_omega[0])
                    intercept = recent_phase[-1] - slope * recent_log_omega[-1]
                    
                    log_target_freq = np.log10(target_freq)
                    estimated_phase = slope * log_target_freq + intercept
                    
                    return estimated_phase
                else:
                    return phase_vals[-1]
                    
        except Exception:
            return -180
    
    def _show_individual_system_analysis(self, system_data, show_margins, show_characteristics):
        """Zeige Analyse der Einzelsysteme"""
        if not system_data:
            return
        
        if show_characteristics:
            st.markdown("---")
            st.markdown("### 📊 Systemcharakteristik-Vergleich")
            
            # Tabelle mit Systemvergleich
            comparison_data = []
            for data in system_data:
                try:
                    dc_gain_db = data['magnitude_db'][0]
                    dc_gain_linear = 10**(dc_gain_db/20)
                    max_gain_db = np.max(data['magnitude_db'])
                    max_gain_freq = data['omega_clean'][np.argmax(data['magnitude_db'])]
                    
                    comparison_data.append({
                        'System': data['label'],
                        'DC-Verstärkung [dB]': f"{dc_gain_db:.1f}",
                        'DC-Verstärkung [linear]': f"{dc_gain_linear:.3f}",
                        'Max. Verstärkung [dB]': f"{max_gain_db:.1f}",
                        'Resonanzfrequenz [rad/s]': f"{max_gain_freq:.3f}"
                    })
                except:
                    comparison_data.append({
                        'System': data['label'],
                        'DC-Verstärkung [dB]': "N/A",
                        'DC-Verstärkung [linear]': "N/A",
                        'Max. Verstärkung [dB]': "N/A",
                        'Resonanzfrequenz [rad/s]': "N/A"
                    })
            
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
        
        if show_margins:
            st.markdown("---")
            st.markdown("### 📏 Stabilitätsreserven-Vergleich")
            
            # Tabelle mit Stabilitätsreserven
            margins_data = []
            for data in system_data:
                try:
                    magnitude_db = data['magnitude_db']
                    phase_deg = data['phase_deg']
                    omega_clean = data['omega_clean']
                    G_s = data.get('G_s')  # Falls verfügbar
                    
                    # Amplitudenreserve
                    phase_shifted = phase_deg + 180
                    phase_crossings = np.where(np.diff(np.signbit(phase_shifted)))[0]
                    
                    gain_margin = "N/A"
                    if len(phase_crossings) > 0:
                        idx = phase_crossings[0]
                        if idx < len(omega_clean) - 1:
                            freq_idx = np.argmin(np.abs(omega_clean - np.interp(0, [phase_shifted[idx], phase_shifted[idx+1]], 
                                                                               [omega_clean[idx], omega_clean[idx+1]])))
                            gain_margin = f"{-magnitude_db[freq_idx]:.1f} dB"
                    elif G_s is not None:
                        # Knickfrequenz-basierte Schätzung falls kein exakter Durchgang
                        s_sym = sp.symbols('s')
                        break_frequencies = self._extract_break_frequencies(G_s, s_sym)
                        asymptotic_phase = self._calculate_asymptotic_phase_from_breaks(G_s, s_sym, break_frequencies)
                        
                        if asymptotic_phase <= -180:
                            estimated_phase_crossing = self._estimate_phase_crossing_from_breaks(
                                G_s, s_sym, break_frequencies, -180
                            )
                            if estimated_phase_crossing:
                                estimated_mag = self._calculate_magnitude_at_frequency(
                                    G_s, s_sym, sp.symbols('omega'), estimated_phase_crossing
                                )
                                gain_margin = f"{-estimated_mag:.1f} dB*"
                            else:
                                gain_margin = "∞ (stabil)"
                        else:
                            gain_margin = "∞ (nie -180°)"
                    
                    # Phasenreserve
                    zero_crossings = np.where(np.diff(np.signbit(magnitude_db)))[0]
                    phase_margin = "N/A"
                    if len(zero_crossings) > 0:
                        idx = zero_crossings[0]
                        if idx < len(omega_clean) - 1:
                            freq_idx = np.argmin(np.abs(omega_clean - np.interp(0, [magnitude_db[idx], magnitude_db[idx+1]], 
                                                                               [omega_clean[idx], omega_clean[idx+1]])))
                            phase_at_crossover = phase_deg[freq_idx]
                            phase_margin = f"{180 + phase_at_crossover:.1f}°"
                    elif G_s is not None:
                        # Knickfrequenz-basierte Schätzung falls kein exakter Durchgang
                        s_sym = sp.symbols('s')
                        break_frequencies = self._extract_break_frequencies(G_s, s_sym)
                        estimated_gain_crossing = self._estimate_gain_crossing_from_breaks(
                            G_s, s_sym, break_frequencies, 0
                        )
                        if estimated_gain_crossing:
                            estimated_phase = self._calculate_phase_at_frequency(
                                G_s, s_sym, sp.symbols('omega'), estimated_gain_crossing
                            )
                            phase_margin = f"{180 + estimated_phase:.1f}°*"
                        else:
                            dc_gain = self._get_dc_gain(G_s, s_sym)
                            if dc_gain and dc_gain > 1:
                                phase_margin = "Hohe DC-Verst."
                            else:
                                phase_margin = "Niedrige Verst."
                    
                    margins_data.append({
                        'System': data['label'],
                        'Amplitudenreserve': gain_margin,
                        'Phasenreserve': phase_margin
                    })
                except:
                    margins_data.append({
                        'System': data['label'],
                        'Amplitudenreserve': "Fehler",
                        'Phasenreserve': "Fehler"
                    })
            
            import pandas as pd
            df_margins = pd.DataFrame(margins_data)
            st.dataframe(df_margins, use_container_width=True)
            
            # Erklärung für Knickfrequenz-basierte Werte
            if any('*' in str(row.get('Amplitudenreserve', '')) or '*' in str(row.get('Phasenreserve', '')) for row in margins_data):
                st.info("ℹ️ Werte mit * basieren auf Knickfrequenz-Analyse (System erreicht asymptotisch kritische Werte)")
    
    def _calculate_stability_margins(self, G_s, s, omega):
        """Berechne Stabilitätsreserven"""
        # Diese Methode wird jetzt von _stability_margins_analysis ersetzt
        st.info("🔗 Detaillierte Stabilitätsreserven finden Sie im Stabilitäts-Tab")
    
    def _extract_break_frequencies(self, G_s, s):
        """Extrahiere Knickfrequenzen aus Übertragungsfunktion"""
        try:
            break_freqs = []
            break_types = []
            
            # Pole verarbeiten
            den = sp.denom(G_s)
            poles = sp.solve(den, s)
            if not poles:
                # Fallback: numerische Methode
                den_poly = sp.Poly(den, s)
                coeffs = [float(c) for c in den_poly.all_coeffs()]
                if len(coeffs) > 1:
                    numerical_poles = np.roots(coeffs)
                    poles = [sp.sympify(complex(p)) for p in numerical_poles]
            
            for pole in poles:
                pole_val = complex(pole.evalf())
                if abs(pole_val.imag) < 1e-10 and pole_val.real < 0:
                    # Reeller Pol: ωk = |Re(p)|
                    break_freqs.append(abs(pole_val.real))
                    break_types.append('pole')
                elif pole_val.real < 0:
                    # Komplexer Pol: ωk = |p|
                    break_freqs.append(abs(pole_val))
                    break_types.append('complex_pole')
            
            # Nullstellen verarbeiten
            num = sp.numer(G_s)
            zeros = sp.solve(num, s)
            if not zeros:
                # Fallback: numerische Methode
                num_poly = sp.Poly(num, s)
                coeffs = [float(c) for c in num_poly.all_coeffs()]
                if len(coeffs) > 1:
                    numerical_zeros = np.roots(coeffs)
                    zeros = [sp.sympify(complex(z)) for z in numerical_zeros]
            
            for zero in zeros:
                zero_val = complex(zero.evalf())
                if abs(zero_val.imag) < 1e-10:
                    # Reelle Nullstelle: ωk = |Re(z)|
                    break_freqs.append(abs(zero_val.real))
                    break_types.append('zero')
                else:
                    # Komplexe Nullstelle: ωk = |z|
                    break_freqs.append(abs(zero_val))
                    break_types.append('complex_zero')
            
            # Sortiere nach Frequenz
            sorted_data = sorted(zip(break_freqs, break_types))
            return sorted_data
            
        except Exception as e:
            st.error(f"Fehler bei Knickfrequenz-Extraktion: {e}")
            return []
    
    def _evaluate_stability_from_break_frequencies(self, break_freqs, G_s, s):
        """Bewerte Stabilität basierend auf Knickfrequenzen"""
        try:
            # DC-Verstärkung bestimmen
            try:
                K_dc = float(G_s.subs(s, 0))
                dc_gain_db = 20 * np.log10(abs(K_dc)) if K_dc != 0 else -float('inf')
            except:
                dc_gain_db = 0  # Fallback
            
            # Systemordnung bestimmen
            num_degree = sp.degree(sp.numer(G_s), s)
            den_degree = sp.degree(sp.denom(G_s), s)
            system_order = int(den_degree) - int(num_degree) if den_degree and num_degree else 1
            
            # Hochfrequenz-Asymptote: -20n dB/Dekade
            hf_slope = -20 * system_order
            
            st.write(f"**Systemanalyse:**")
            st.write(f"• DC-Verstärkung: {dc_gain_db:.1f} dB")
            st.write(f"• Systemordnung: {system_order}")
            st.write(f"• HF-Asymptote: {hf_slope} dB/Dekade")
            
            # Abschätzung der kritischen Frequenz
            if break_freqs and dc_gain_db > 0:
                # Grobe Abschätzung: Bei welcher Frequenz erreicht System 0dB?
                # Unter Annahme, dass jede Dekade um 20 dB fällt
                decades_to_0db = dc_gain_db / 20
                first_break_freq = break_freqs[0][0]
                estimated_0db_freq = first_break_freq * (10 ** decades_to_0db)
                
                st.write(f"**Kritische Frequenz (geschätzt):** ~{estimated_0db_freq:.2f} rad/s")
                
                # Bewertung basierend auf Frequenzabstand
                freq_separation = min([freq for freq, _ in break_freqs[1:]] + [float('inf')]) / break_freqs[0][0] if len(break_freqs) > 1 else float('inf')
                
                if freq_separation > 10:
                    st.success("✅ Gute Frequenztrennung (> 1 Dekade)")
                elif freq_separation > 3:
                    st.warning("⚠️ Moderate Frequenztrennung")
                else:
                    st.error("❌ Schlechte Frequenztrennung (< 0.5 Dekaden)")
            
        except Exception as e:
            st.error(f"Fehler bei Knickfrequenz-Bewertung: {e}")
    
    def _estimate_phase_margin_from_structure(self, G_s, s):
        """Schätze Phasenreserve basierend auf Systemstruktur"""
        try:
            # Pole und Nullstellen analysieren
            poles = sp.solve(sp.denom(G_s), s)
            zeros = sp.solve(sp.numer(G_s), s)
            
            # Robuste Fallback-Methode
            if not poles:
                den_poly = sp.Poly(sp.denom(G_s), s)
                coeffs = [float(c) for c in den_poly.all_coeffs()]
                if len(coeffs) > 1:
                    numerical_poles = np.roots(coeffs)
                    poles = [sp.sympify(complex(p)) for p in numerical_poles]
            
            if not zeros:
                num_poly = sp.Poly(sp.numer(G_s), s)
                coeffs = [float(c) for c in num_poly.all_coeffs()]
                if len(coeffs) > 1:
                    numerical_zeros = np.roots(coeffs)
                    zeros = [sp.sympify(complex(z)) for z in numerical_zeros]
            
            # Grobe Phasenreserve-Schätzung
            total_phase = 0
            
            # Beitrag der Pole (jeweils -90° bei hohen Frequenzen)
            stable_poles = [p for p in poles if complex(p.evalf()).real < 0]
            total_phase -= len(stable_poles) * 90
            
            # Beitrag der Nullstellen (jeweils +90° bei hohen Frequenzen)
            stable_zeros = [z for z in zeros if complex(z.evalf()).real < 0]
            total_phase += len(stable_zeros) * 90
            
            # Startphase (bei niedrigen Frequenzen ist Phase meist 0°)
            estimated_phase_margin = 180 + total_phase
            
            return estimated_phase_margin
            
        except Exception:
            return None
