"""
Frequenzgang-Analyse Modul
Spezialisiert auf umfassende Frequenzgang-Berichte und komplexe Übertragungsfunktionen
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

class FrequencyResponseModule(BaseModule):
    """Modul für umfassende Frequenzgang-Analyse mit komplexen Übertragungsfunktionen"""
    
    def __init__(self):
        super().__init__(
            "Frequenzgang-Analyse",
            "Umfassende Analyse von Übertragungsfunktionen im Frequenzbereich mit automatischen Berichten"
        )
    
    def render(self):
        self.display_description()
        
        tab1, tab2, tab3 = st.tabs([
            "Komplexe Übertragungsfunktion",
            "Frequenzgang-Bericht",
            "Vergleichsanalyse"
        ])
        
        with tab1:
            self.complex_transfer_function()
        
        with tab2:
            self.comprehensive_report()
        
        with tab3:
            self.comparison_analysis()
    
    def complex_transfer_function(self):
        """Eingabe und Analyse komplexer Übertragungsfunktionen"""
        st.subheader("Komplexe Übertragungsfunktion")
        
        st.markdown("""
        **Eingabeoptionen:**
        1. **Standard G(s)** - Wird automatisch zu G(jω) konvertiert
        2. **Direkt G(jw)** - Bereits im Frequenzbereich
        3. **Numerische Werte** - Für spezifische Frequenzen
        """)
        
        st.info("ℹ️ **Automatische Multiplikation:** Das System erkennt automatisch Ausdrücke wie `2s`, `(s+1)(s+2)` oder `s(s+1)` und fügt `*` hinzu. In seltenen Fällen kann es nötig sein, `*` manuell zu setzen.")
        
        input_type = st.selectbox(
            "Eingabetyp wählen:",
            ["Standard G(s)", "Direkt G(jw)", "Numerische Werte"]
        )
        
        if input_type == "Standard G(s)":
            self._standard_transfer_function()
        elif input_type == "Direkt G(jw)":
            self._direct_frequency_domain()
        else:
            self._numerical_values()
    
    def _standard_transfer_function(self):
        """Standard Übertragungsfunktion G(s)"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            numerator = st.text_input(
                "Zähler (Numerator):",
                value="1",
                help="Beispiel: s+1, s**2+2s+1 (automatische Multiplikation: 2s = 2*s)"
            )
            
            denominator = st.text_input(
                "Nenner (Denominator):",
                value="s**2+2*s+1",
                help="Beispiel: s+1, (s+1)(s+2) (automatische Multiplikation)"
            )
        
        with col2:
            st.markdown("**Einstellungen:**")
            auto_convert = st.checkbox("Auto zu G(jω) konvertieren", value=True)
            show_steps = st.checkbox("Umwandlungsschritte zeigen", value=True)
        
        if st.button("Analysieren", key="analyze_standard"):
            with st.spinner("Analysiere Übertragungsfunktion..."):
                try:
                    # SymPy Symbole
                    s, omega = sp.symbols('s omega', real=True)
                    j = sp.I
                    symbols_dict = {'s': s, 'omega': omega}
                    
                    # Debug Info
                    st.info(f"Eingabe: Zähler='{numerator}', Nenner='{denominator}'")
                    
                    # Übertragungsfunktion erstellen
                    num_expr = safe_sympify(numerator, symbols_dict)
                    den_expr = safe_sympify(denominator, symbols_dict)
                    st.success(f"Parsing erfolgreich: {num_expr} / {den_expr}")
                        
                    G_s = num_expr / den_expr
                    
                    st.latex(f"G(s) = \\frac{{{sp.latex(num_expr)}}}{{{sp.latex(den_expr)}}}")
                    
                    if auto_convert:
                        if show_steps:
                            st.markdown("### 🔄 Umwandlungsschritte s → jω")
                            
                            steps = [
                                ("Ausgangspunkt", f"G(s) = {sp.latex(G_s)}"),
                                ("Substitution", "s \\rightarrow j\\omega"),
                                ("Einsetzen", f"G(j\\omega) = {sp.latex(G_s.subs(s, j*omega))}"),
                            ]
                            
                            # Verwende die lokale Fallback-Funktion für Tupel-Format
                            for step_name, step_content in steps:
                                st.markdown(f"**{step_name}:**")
                                st.latex(step_content)
                        
                        # G(jω) berechnen
                        G_jw = G_s.subs(s, j*omega)
                        
                        st.markdown("### 📊 Frequenzbereichsdarstellung")
                        st.latex(f"G(j\\omega) = {sp.latex(G_jw)}")
                        
                        # Real- und Imaginärteil extrahieren
                        G_jw_expanded = sp.expand_complex(G_jw, deep=True)
                        real_part = sp.re(G_jw_expanded)
                        imag_part = sp.im(G_jw_expanded)
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown("**Realteil:**")
                            st.latex(f"\\text{{Re}}[G(j\\omega)] = {sp.latex(real_part)}")
                        
                        with col_b:
                            st.markdown("**Imaginärteil:**")
                            st.latex(f"\\text{{Im}}[G(j\\omega)] = {sp.latex(imag_part)}")
                        
                        # Amplituden- und Phasengang
                        magnitude = sp.sqrt(real_part**2 + imag_part**2)
                        phase = sp.atan2(imag_part, real_part)
                        
                        st.markdown("### 📈 Amplituden- und Phasengang")
                        
                        col_c, col_d = st.columns(2)
                        
                        with col_c:
                            st.markdown("**Amplitudengang:**")
                            st.latex(f"|G(j\\omega)| = {sp.latex(magnitude)}")
                        
                        with col_d:
                            st.markdown("**Phasengang:**")
                            st.latex(f"\\angle G(j\\omega) = {sp.latex(phase)}")
                        
                        # Automatische Frequenzbereichsdiagramme erstellen
                        try:
                            self._create_frequency_plots(G_s, real_part, imag_part, magnitude, phase, omega)
                        except Exception as plot_error:
                            st.error(f"Fehler bei der Plot-Erstellung: {plot_error}")
                    
                except Exception as e:
                    st.error(f"Fehler bei der Analyse: {e}")
                    st.info("Tipp: Sie können direkt 2s statt 2*s schreiben. Potenzen mit ** eingeben.")
                    # Zusätzliche Debug-Info
                    import traceback
                    with st.expander("Debug-Informationen"):
                        st.code(traceback.format_exc())
    
    def _direct_frequency_domain(self):
        """Direkte Eingabe im Frequenzbereich"""
        st.markdown("**Direkteingabe G(jω):**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            real_input = st.text_input(
                "Realteil Re[G(jω)]:",
                value="1/(1-omega**2)",
                help="Funktion von ω. Beispiel: 1/(1-omega**2)"
            )
            
            imag_input = st.text_input(
                "Imaginärteil Im[G(jω)]:",
                value="-omega/(1-omega**2)",
                help="Funktion von ω. Beispiel: -omega/(1-omega**2)"
            )
        
        with col2:
            st.markdown("**Frequenzbereich:**")
            omega_min = st.number_input("ω min [rad/s]:", value=0.1, format="%.2f")
            omega_max = st.number_input("ω max [rad/s]:", value=10.0, format="%.1f")
            
        if st.button("Analysieren", key="analyze_direct"):
            try:
                omega = sp.Symbol('omega', real=True)
                
                # Ausdrücke parsen
                real_expr = safe_sympify(real_input, {'omega': omega})
                imag_expr = safe_sympify(imag_input, {'omega': omega})
                
                st.markdown("### 🎯 Eingegebene Frequenzantwort")
                st.latex(f"G(j\\omega) = {sp.latex(real_expr)} + j({sp.latex(imag_expr)})")
                
                # Magnitude und Phase berechnen
                magnitude = sp.sqrt(real_expr**2 + imag_expr**2)
                phase = sp.atan2(imag_expr, real_expr)
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Amplitudengang:**")
                    st.latex(f"|G(j\\omega)| = {sp.latex(magnitude)}")
                
                with col_b:
                    st.markdown("**Phasengang:**")
                    st.latex(f"\\angle G(j\\omega) = {sp.latex(phase)}")
                
                # Plots erstellen
                self._create_frequency_plots_from_expressions(
                    real_expr, imag_expr, magnitude, phase, omega, omega_min, omega_max
                )
                
            except Exception as e:
                st.error(f"Fehler bei der Analyse: {e}")
    
    def _numerical_values(self):
        """Numerische Werte für spezifische Frequenzen"""
        st.markdown("**Numerische Eingabe für spezifische Frequenzen:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            freq_input = st.text_area(
                "Frequenzen [rad/s] (eine pro Zeile):",
                value="0.1\\n1\\n10\\n100",
                height=100
            )
        
        with col2:
            values_input = st.text_area(
                "G(jω) Werte (Real+j*Imag, eine pro Zeile):",
                value="1+0j\\n0.5-0.5j\\n0.1-0.9j\\n0.01-0.99j",
                height=100
            )
        
        if st.button("Analysieren", key="analyze_numerical"):
            try:
                # Frequenzen parsen
                freq_lines = freq_input.strip().split('\\n')
                frequencies = [float(line.strip()) for line in freq_lines if line.strip()]
                
                # Komplexe Werte parsen
                value_lines = values_input.strip().split('\\n')
                complex_values = []
                
                for line in value_lines:
                    if line.strip():
                        # Einfache Parsing für komplexe Zahlen
                        line = line.strip().replace('i', 'j').replace('I', 'j')
                        complex_values.append(complex(line))
                
                if len(frequencies) != len(complex_values):
                    st.error("Anzahl der Frequenzen muss gleich der Anzahl der Werte sein!")
                    return
                
                # Daten anzeigen
                st.markdown("### 📊 Eingabedaten")
                
                data_df = {
                    'ω [rad/s]': frequencies,
                    'Re[G(jω)]': [val.real for val in complex_values],
                    'Im[G(jω)]': [val.imag for val in complex_values],
                    '|G(jω)|': [abs(val) for val in complex_values],
                    '∠G(jω) [°]': [np.angle(val)*180/np.pi for val in complex_values]
                }
                
                st.dataframe(data_df, use_container_width=True)
                
                # Plots erstellen
                self._create_plots_from_numerical_data(frequencies, complex_values)
                
            except Exception as e:
                st.error(f"Fehler bei der Analyse: {e}")
                st.info("Tipp: Verwenden Sie das Format 'a+bj' für komplexe Zahlen")
    
    def _create_frequency_plots(self, G_s, real_part, imag_part, magnitude, phase, omega_sym):
        """Erstelle Frequenzbereichsdiagramme aus symbolischen Ausdrücken"""
        
        # Sicherheitsabfrage
        if not hasattr(omega_sym, 'is_Symbol') or not omega_sym.is_Symbol:
            raise TypeError(f"omega_sym muss ein SymPy Symbol sein, erhalten: {type(omega_sym)}")
        
        # Frequenzvektor
        omega_vals = np.logspace(-1, 2, 1000)
        
        try:
            # Numerische Auswertung
            real_func = sp.lambdify(omega_sym, real_part, 'numpy')
            imag_func = sp.lambdify(omega_sym, imag_part, 'numpy')
            mag_func = sp.lambdify(omega_sym, magnitude, 'numpy')
            phase_func = sp.lambdify(omega_sym, phase, 'numpy')
            
            real_vals = real_func(omega_vals)
            imag_vals = imag_func(omega_vals)
            mag_vals = mag_func(omega_vals)
            phase_vals = phase_func(omega_vals) * 180 / np.pi
            
            # Filter out infinite/NaN values
            finite_mask = np.isfinite(real_vals) & np.isfinite(imag_vals)
            
            omega_clean = omega_vals[finite_mask]
            real_clean = real_vals[finite_mask]
            imag_clean = imag_vals[finite_mask]
            mag_clean = mag_vals[finite_mask]
            phase_clean = phase_vals[finite_mask]
            
            self._create_comprehensive_plots(
                omega_clean, real_clean, imag_clean, mag_clean, phase_clean
            )
            
        except Exception as e:
            st.error(f"Fehler bei der Plot-Erstellung: {e}")
    
    def _create_frequency_plots_from_expressions(self, real_expr, imag_expr, magnitude, phase, omega_sym, omega_min, omega_max):
        """Erstelle Plots aus direkten Frequenzbereichsausdrücken"""
        
        omega_vals = np.logspace(np.log10(omega_min), np.log10(omega_max), 1000)
        
        try:
            real_func = sp.lambdify(omega_sym, real_expr, 'numpy')
            imag_func = sp.lambdify(omega_sym, imag_expr, 'numpy')
            mag_func = sp.lambdify(omega_sym, magnitude, 'numpy')
            phase_func = sp.lambdify(omega_sym, phase, 'numpy')
            
            real_vals = real_func(omega_vals)
            imag_vals = imag_func(omega_vals)
            mag_vals = mag_func(omega_vals)
            phase_vals = phase_func(omega_vals) * 180 / np.pi
            
            # Filter out infinite/NaN values
            finite_mask = np.isfinite(real_vals) & np.isfinite(imag_vals)
            
            omega_clean = omega_vals[finite_mask]
            real_clean = real_vals[finite_mask]
            imag_clean = imag_vals[finite_mask]
            mag_clean = mag_vals[finite_mask]
            phase_clean = phase_vals[finite_mask]
            
            self._create_comprehensive_plots(
                omega_clean, real_clean, imag_clean, mag_clean, phase_clean
            )
            
        except Exception as e:
            st.error(f"Fehler bei der Plot-Erstellung: {e}")
    
    def _create_plots_from_numerical_data(self, frequencies, complex_values):
        """Erstelle Plots aus numerischen Daten"""
        
        omega_vals = np.array(frequencies)
        real_vals = np.array([val.real for val in complex_values])
        imag_vals = np.array([val.imag for val in complex_values])
        mag_vals = np.array([abs(val) for val in complex_values])
        phase_vals = np.array([np.angle(val)*180/np.pi for val in complex_values])
        
        self._create_comprehensive_plots(
            omega_vals, real_vals, imag_vals, mag_vals, phase_vals, is_discrete=True
        )
    
    def _create_comprehensive_plots(self, omega, real_vals, imag_vals, mag_vals, phase_vals, is_discrete=False):
        """Erstelle umfassende Frequenzgang-Diagramme"""
        
        st.markdown("### 📊 Frequenzgang-Diagramme")
        
        # 1. Re-Im Diagramm (Nyquist-ähnlich)
        st.markdown("#### 🎯 Real-Imaginär-Diagramm (automatisch skaliert)")
        
        fig_nyquist = go.Figure()
        
        mode = 'markers' if is_discrete else 'lines'
        
        fig_nyquist.add_trace(go.Scatter(
            x=real_vals, y=imag_vals,
            mode=mode if is_discrete else 'lines+markers',
            name='G(jω)',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            hovertemplate='ω: %{text}<br>Re: %{x:.3f}<br>Im: %{y:.3f}<extra></extra>',
            text=[f'{w:.2f}' for w in omega]
        ))
        
        # Koordinatenachsen
        fig_nyquist.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        fig_nyquist.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
        
        # Automatische Skalierung mit Padding
        x_range = [min(real_vals), max(real_vals)]
        y_range = [min(imag_vals), max(imag_vals)]
        x_padding = (x_range[1] - x_range[0]) * 0.1
        y_padding = (y_range[1] - y_range[0]) * 0.1
        
        # Kritischer Punkt -1+0j hinzufügen (falls relevant)
        if x_range[0] <= -1 <= x_range[1] and y_range[0] <= 0 <= y_range[1]:
            fig_nyquist.add_trace(go.Scatter(
                x=[-1], y=[0],
                mode='markers',
                marker=dict(symbol='x', size=15, color='red', line=dict(width=3)),
                name='Kritischer Punkt (-1+0j)',
                hovertemplate='Kritischer Punkt<br>Re: -1<br>Im: 0<extra></extra>'
            ))
        
        fig_nyquist.update_layout(
            title="Real-Imaginär-Diagramm",
            xaxis_title="Realteil",
            yaxis_title="Imaginärteil",
            showlegend=True,
            width=700, height=500,
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
        
        st.plotly_chart(fig_nyquist, use_container_width=True)
        
        # 2. Bode-Diagramm
        st.markdown("#### 📈 Bode-Diagramm")
        
        fig_bode = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Amplitudengang', 'Phasengang'),
            vertical_spacing=0.1
        )
        
        # Amplitudengang in dB
        mag_db = 20 * np.log10(np.abs(mag_vals))
        
        fig_bode.add_trace(
            go.Scatter(
                x=omega, y=mag_db, 
                mode=mode,
                name='|G(jω)| [dB]',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Phasengang
        fig_bode.add_trace(
            go.Scatter(
                x=omega, y=phase_vals, 
                mode=mode,
                name='∠G(jω) [°]',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Referenzlinien
        fig_bode.add_hline(y=0, line_dash="dash", line_color="gray", 
                          annotation_text="0 dB", row=1, col=1)
        fig_bode.add_hline(y=-180, line_dash="dash", line_color="gray",
                          annotation_text="-180°", row=2, col=1)
        
        fig_bode.update_xaxes(type="log", title_text="Frequenz ω [rad/s]")
        fig_bode.update_yaxes(title_text="Magnitude [dB]", row=1, col=1)
        fig_bode.update_yaxes(title_text="Phase [°]", row=2, col=1)
        
        fig_bode.update_layout(
            title="Bode-Diagramm",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig_bode, use_container_width=True)
        
        # 3. 3D-Darstellung (optional)
        if st.checkbox("3D-Darstellung anzeigen"):
            self._create_3d_plot(omega, real_vals, imag_vals, mag_vals)
    
    def _create_3d_plot(self, omega, real_vals, imag_vals, mag_vals):
        """Erstelle 3D-Darstellung des Frequenzgangs"""
        
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=real_vals,
            y=imag_vals, 
            z=omega,
            mode='lines+markers',
            line=dict(color=mag_vals, colorscale='Viridis', width=4),
            marker=dict(size=3, color=mag_vals, colorscale='Viridis'),
            hovertemplate='ω: %{z:.2f}<br>Re: %{x:.3f}<br>Im: %{y:.3f}<br>|G|: %{marker.color:.3f}<extra></extra>'
        )])
        
        fig_3d.update_layout(
            title="3D Frequenzgang-Darstellung",
            scene=dict(
                xaxis_title="Realteil",
                yaxis_title="Imaginärteil",
                zaxis_title="Frequenz ω [rad/s]"
            ),
            width=700, height=600
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    def comprehensive_report(self):
        """Umfassender Frequenzgang-Bericht"""
        st.subheader("📋 Umfassender Frequenzgang-Bericht")
        
        st.markdown("""
        Generiert einen detaillierten Bericht über die Frequenzgang-Eigenschaften
        einer Übertragungsfunktion mit automatischer Analyse aller wichtigen Parameter.
        """)
        
        # Eingabe für Bericht
        transfer_function = st.text_input(
            "Übertragungsfunktion G(s):",
            value="10/((s+1)*(s+5))",
            help="Beispiele: 1/(s+1), K/(s^2+2*s+1)"
        )
        
        if st.button("📊 Detaillierten Bericht erstellen"):
            try:
                self._generate_comprehensive_report(transfer_function)
            except Exception as e:
                st.error(f"Fehler bei der Berichtsgenerierung: {e}")
    
    def _generate_comprehensive_report(self, tf_string):
        """Generiere umfassenden Frequenzgang-Bericht"""
        
        s, omega = sp.symbols('s omega', real=True)
        j = sp.I
        
        # Übertragungsfunktion parsen
        G_s = safe_sympify(tf_string, {'s': s, 'omega': omega})
        
        st.markdown("## 📊 FREQUENZGANG-ANALYSE BERICHT")
        st.markdown("---")
        
        # 1. Systemidentifikation
        st.markdown("### 1️⃣ Systemidentifikation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Übertragungsfunktion:**")
            st.latex(f"G(s) = {sp.latex(G_s)}")
        
        with col2:
            # Systemordnung bestimmen
            num_degree = sp.degree(sp.numer(G_s), s)
            den_degree = sp.degree(sp.denom(G_s), s)
            
            st.markdown("**Systemeigenschaften:**")
            st.write(f"• Zählergrad: {num_degree}")
            st.write(f"• Nennergrad: {den_degree}")
            st.write(f"• Systemordnung: {den_degree}")
            st.write(f"• Systemtyp: {'Eigentlich' if num_degree <= den_degree else 'Unecht'}")
        
        # 2. Umwandlung in Frequenzbereich
        st.markdown("### 2️⃣ Umwandlung s → jω")
        
        G_jw = G_s.subs(s, j*omega)
        G_jw_expanded = sp.expand_complex(G_jw, deep=True)
        
        real_part = sp.re(G_jw_expanded)
        imag_part = sp.im(G_jw_expanded)
        
        st.latex(f"G(j\\omega) = {sp.latex(G_jw)}")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Realteil:**")
            st.latex(f"\\text{{Re}}[G(j\\omega)] = {sp.latex(real_part)}")
        
        with col_b:
            st.markdown("**Imaginärteil:**")
            st.latex(f"\\text{{Im}}[G(j\\omega)] = {sp.latex(imag_part)}")
        
        # 3. Charakteristische Frequenzen
        st.markdown("### 3️⃣ Charakteristische Frequenzen")
        
        # Pole und Nullstellen
        poles = sp.solve(sp.denom(G_s), s)
        zeros = sp.solve(sp.numer(G_s), s)
        
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.markdown("**Pole (Eckfrequenzen):**")
            for i, pole in enumerate(poles):
                pole_val = complex(pole.evalf())
                if abs(pole_val.imag) < 1e-10 and pole_val.real < 0:
                    omega_pole = abs(pole_val.real)
                    st.write(f"ω{i+1} = {omega_pole:.3f} rad/s")
                elif pole_val.real < 0:
                    omega_n = abs(pole_val)
                    zeta = -pole_val.real / omega_n
                    st.write(f"ωₙ{i+1} = {omega_n:.3f} rad/s (ζ = {zeta:.3f})")
        
        with col_y:
            st.markdown("**Nullstellen:**")
            if zeros:
                for i, zero in enumerate(zeros):
                    zero_val = complex(zero.evalf())
                    if abs(zero_val.imag) < 1e-10:
                        omega_zero = abs(zero_val.real)
                        st.write(f"ω{i+1} = {omega_zero:.3f} rad/s")
            else:
                st.write("Keine endlichen Nullstellen")
        
        # 4. Frequenzgang-Plots
        st.markdown("### 4️⃣ Grafische Darstellung")
        
        # Automatische Frequenzbereichsbestimmung
        pole_freqs = []
        for pole in poles:
            pole_val = complex(pole.evalf())
            if pole_val.real < 0:
                pole_freqs.append(abs(pole_val))
        
        if pole_freqs:
            omega_min = min(pole_freqs) / 100
            omega_max = max(pole_freqs) * 100
        else:
            omega_min = 0.01
            omega_max = 100
        
        # Plot erstellen
        self._create_frequency_plots(G_s, real_part, imag_part, 
                                   sp.sqrt(real_part**2 + imag_part**2),
                                   sp.atan2(imag_part, real_part), omega)
        
        # 5. Stabilitätsanalyse
        st.markdown("### 5️⃣ Stabilitätsanalyse")
        
        stable = all(complex(pole.evalf()).real < 0 for pole in poles)
        
        if stable:
            st.success("✅ **System ist stabil** - Alle Pole in der linken Halbebene")
        else:
            st.error("❌ **System ist instabil** - Mindestens ein Pol in der rechten Halbebene")
        
        # Zeitkonstanten
        st.markdown("**Zeitkonstanten:**")
        for i, pole in enumerate(poles):
            pole_val = complex(pole.evalf())
            if abs(pole_val.imag) < 1e-10 and pole_val.real < 0:
                tau = -1/pole_val.real
                st.write(f"τ{i+1} = {tau:.3f} s")
        
        # Stabilitätsreserven
        st.markdown("**Stabilitätsreserven:**")
        try:
            self._calculate_stability_margins_single(G_s, s, omega)
        except Exception as e:
            st.write(f"Stabilitätsreserven konnten nicht berechnet werden: {e}")
        
        # 6. Frequenzgang-Eigenschaften
        st.markdown("### 6️⃣ Frequenzgang-Eigenschaften")
        
        # DC-Verstärkung
        try:
            dc_gain = float(G_s.subs(s, 0))
            st.write(f"**DC-Verstärkung:** K₀ = {dc_gain:.3f}")
            st.write(f"**DC-Verstärkung [dB]:** {20*np.log10(abs(dc_gain)):.1f} dB")
        except:
            st.write("**DC-Verstärkung:** Nicht definiert (Pol bei s=0)")
        
        # Hochfrequenzverhalten
        try:
            if num_degree < den_degree:
                st.write(f"**Hochfrequenzverhalten:** |G(j∞)| = 0 (Steigung: {-20*(den_degree-num_degree)} dB/Dek)")
            elif num_degree == den_degree:
                hf_gain = float((sp.numer(G_s).as_leading_term(s) / sp.denom(G_s).as_leading_term(s)))
                st.write(f"**Hochfrequenzverhalten:** |G(j∞)| = {abs(hf_gain):.3f}")
            else:
                st.write("**Hochfrequenzverhalten:** |G(j∞)| = ∞ (System unecht)")
        except:
            st.write("**Hochfrequenzverhalten:** Analyse nicht möglich")
        
        # 7. Zusammenfassung
        st.markdown("### 7️⃣ Zusammenfassung")
        
        summary_points = [
            f"System {den_degree}. Ordnung",
            "Stabil" if stable else "Instabil",
            f"{len(poles)} Pole, {len(zeros)} endliche Nullstellen",
            "Eigentlich" if num_degree <= den_degree else "Unecht"
        ]
        
        for point in summary_points:
            st.write(f"• {point}")
    
    def comparison_analysis(self):
        """Vergleichsanalyse mehrerer Übertragungsfunktionen"""
        st.subheader("🔄 Vergleichsanalyse")
        
        st.markdown("""
        Vergleichen Sie mehrere Übertragungsfunktionen hinsichtlich
        ihrer Frequenzgang-Eigenschaften.
        """)
        
        # Eingabe für mehrere Systeme
        num_systems = st.number_input("Anzahl Systeme:", min_value=2, max_value=5, value=2)
        
        systems = []
        labels = []
        
        for i in range(num_systems):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                tf = st.text_input(
                    f"System {i+1} G{i+1}(s):",
                    value=f"1/(s+{i+1})",
                    key=f"system_{i}"
                )
                systems.append(tf)
            
            with col2:
                label = st.text_input(
                    f"Label:",
                    value=f"System {i+1}",
                    key=f"label_{i}"
                )
                labels.append(label)
        
        # Zusätzliche Optionen
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            show_combined = st.checkbox("Gesamtübertragungsfunktion anzeigen", 
                                      help="Zeigt das Produkt aller Systeme: G_gesamt = G1 × G2 × ...")
        
        with col_opt2:
            show_stability_margins = st.checkbox("Stabilitätsreserven anzeigen", 
                                                help="Zeigt Amplituden- und Phasenreserve an")
        
        if st.button("🔄 Vergleichsanalyse starten"):
            try:
                self._create_comparison_plots(systems, labels, show_combined, show_stability_margins)
            except Exception as e:
                st.error(f"Fehler beim Vergleich: {e}")
    
    def _create_comparison_plots(self, systems, labels, show_combined=False, show_stability_margins=False):
        """Erstelle Vergleichsdiagramme"""
        
        s, omega = sp.symbols('s omega', real=True)
        j = sp.I
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Frequenzvektor
        omega_vals = np.logspace(-2, 2, 1000)
        
        st.markdown("### 📊 Vergleichsdiagramme")
        
        # 1. Nyquist-Vergleich
        fig_nyquist = go.Figure()
        
        # 2. Bode-Vergleich
        fig_bode = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Amplitudengang Vergleich', 'Phasengang Vergleich'),
            vertical_spacing=0.1
        )
        
        # Daten für Gesamtübertragungsfunktion sammeln
        all_G_vals = []
        valid_systems = []
        
        for i, (tf_str, label) in enumerate(zip(systems, labels)):
            try:
                G_s = safe_sympify(tf_str, {'s': s, 'omega': omega})
                
                # Stelle sicher, dass G_s ein SymPy-Ausdruck ist
                if not hasattr(G_s, 'subs'):
                    # Für konstante Werte (P-Glieder) zu SymPy-Ausdruck konvertieren
                    G_s = sp.sympify(G_s)
                
                G_jw = G_s.subs(s, j*omega)
                
                # Numerische Auswertung
                # Für konstante Werte (P-Glieder) spezielle Behandlung
                if G_jw.is_number:
                    # Konstanter Wert - erstelle Array mit diesem Wert
                    G_vals = np.full_like(omega_vals, complex(G_jw), dtype=complex)
                else:
                    G_func = sp.lambdify(omega, G_jw, 'numpy')
                    G_vals = G_func(omega_vals)
                
                # Filter infinite values
                finite_mask = np.isfinite(G_vals)
                G_clean = G_vals[finite_mask]
                omega_clean = omega_vals[finite_mask]
                
                real_vals = np.real(G_clean)
                imag_vals = np.imag(G_clean)
                mag_vals = np.abs(G_clean)
                phase_vals = np.angle(G_clean) * 180 / np.pi
                
                color = colors[i % len(colors)]
                
                # Nyquist-Plot
                fig_nyquist.add_trace(go.Scatter(
                    x=real_vals, y=imag_vals,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2),
                    hovertemplate=f'{label}<br>Re: %{{x:.3f}}<br>Im: %{{y:.3f}}<extra></extra>'
                ))
                
                # Bode-Plots
                mag_db = 20 * np.log10(mag_vals)
                
                fig_bode.add_trace(
                    go.Scatter(
                        x=omega_clean, y=mag_db,
                        mode='lines',
                        name=label,
                        line=dict(color=color, width=2)
                    ),
                    row=1, col=1
                )
                
                fig_bode.add_trace(
                    go.Scatter(
                        x=omega_clean, y=phase_vals,
                        mode='lines',
                        name=label,
                        line=dict(color=color, width=2),
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                # Daten für Gesamtübertragungsfunktion sammeln
                all_G_vals.append(G_vals)
                valid_systems.append((tf_str, label))
                
            except Exception as e:
                st.warning(f"Fehler bei System {i+1}: {e}")
        
        # Gesamtübertragungsfunktion hinzufügen
        if show_combined and len(all_G_vals) >= 2:
            # Produkt aller Übertragungsfunktionen berechnen
            G_combined = all_G_vals[0].copy()
            for G_vals in all_G_vals[1:]:
                # Sicherstellen, dass beide Arrays gleiche Länge haben
                min_len = min(len(G_combined), len(G_vals))
                G_combined = G_combined[:min_len] * G_vals[:min_len]
            
            omega_combined = omega_vals[:len(G_combined)]
            
            real_combined = np.real(G_combined)
            imag_combined = np.imag(G_combined)
            mag_combined = np.abs(G_combined)
            phase_combined = np.angle(G_combined) * 180 / np.pi
            
            # Gesamtübertragungsfunktion zu Plots hinzufügen
            fig_nyquist.add_trace(go.Scatter(
                x=real_combined, y=imag_combined,
                mode='lines',
                name='Gesamtübertragungsfunktion',
                line=dict(color='black', width=3, dash='dash'),
                hovertemplate='Gesamt<br>Re: %{x:.3f}<br>Im: %{y:.3f}<extra></extra>'
            ))
            
            mag_combined_db = 20 * np.log10(mag_combined)
            
            fig_bode.add_trace(
                go.Scatter(
                    x=omega_combined, y=mag_combined_db,
                    mode='lines',
                    name='Gesamtübertragungsfunktion',
                    line=dict(color='black', width=3, dash='dash')
                ),
                row=1, col=1
            )
            
            fig_bode.add_trace(
                go.Scatter(
                    x=omega_combined, y=phase_combined,
                    mode='lines',
                    name='Gesamtübertragungsfunktion',
                    line=dict(color='black', width=3, dash='dash'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Nyquist-Layout
        fig_nyquist.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        fig_nyquist.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
        
        fig_nyquist.update_layout(
            title="Nyquist-Diagramm Vergleich",
            xaxis_title="Realteil",
            yaxis_title="Imaginärteil",
            showlegend=True,
            width=700, height=500
        )
        
        st.plotly_chart(fig_nyquist, use_container_width=True)
        
        # Bode-Layout
        fig_bode.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig_bode.add_hline(y=-180, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig_bode.update_xaxes(type="log", title_text="Frequenz ω [rad/s]")
        fig_bode.update_yaxes(title_text="Magnitude [dB]", row=1, col=1)
        fig_bode.update_yaxes(title_text="Phase [°]", row=2, col=1)
        
        fig_bode.update_layout(
            title="Bode-Diagramm Vergleich",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig_bode, use_container_width=True)
        
        # Stabilitätsreserven anzeigen
        if show_stability_margins:
            self._analyze_stability_margins(valid_systems, omega_vals, s, omega, j)
        
        # Vergleichstabelle
        st.markdown("### 📋 Systemvergleich")
        
        comparison_data = []
        
        for i, (tf_str, label) in enumerate(zip(systems, labels)):
            try:
                G_s = safe_sympify(tf_str, {'s': s, 'omega': omega})
                
                # Stelle sicher, dass G_s ein SymPy-Ausdruck ist
                if not hasattr(G_s, 'subs'):
                    # Für konstante Werte (P-Glieder) zu SymPy-Ausdruck konvertieren
                    G_s = sp.sympify(G_s)
                
                # DC-Verstärkung
                try:
                    dc_gain_raw = G_s.subs(s, 0)
                    dc_gain = float(dc_gain_raw)
                    dc_gain_db = 20 * np.log10(abs(dc_gain))
                except:
                    dc_gain = "N/A"
                    dc_gain_db = "N/A"
                
                # Systemordnung
                try:
                    den_degree_raw = sp.degree(sp.denom(G_s), s)
                    # Konvertiere SymPy Zero zu Python int
                    den_degree = int(den_degree_raw) if den_degree_raw is not None else 0
                except:
                    # Für konstante Werte ist die Ordnung 0
                    den_degree = 0
                
                # Stabilität
                try:
                    poles = sp.solve(sp.denom(G_s), s)
                    stable = all(complex(pole.evalf()).real < 0 for pole in poles)
                except:
                    # Konstante Werte (P-Glieder) haben keine Pole und sind stabil
                    stable = True
                
                comparison_data.append({
                    'System': label,
                    'G(s)': str(G_s),
                    'Ordnung': den_degree,
                    'DC-Verstärkung': f"{dc_gain:.3f}" if isinstance(dc_gain, (int, float)) else dc_gain,
                    'DC [dB]': f"{dc_gain_db:.1f}" if isinstance(dc_gain_db, (int, float)) else dc_gain_db,
                    'Stabil': "Ja" if stable else "Nein"
                })
                
            except Exception as e:
                comparison_data.append({
                    'System': label,
                    'G(s)': tf_str,
                    'Ordnung': "Fehler",
                    'DC-Verstärkung': "Fehler",
                    'DC [dB]': "Fehler",
                    'Stabil': "Fehler"
                })
        
        st.dataframe(comparison_data, use_container_width=True)
    
    def _analyze_stability_margins(self, systems, omega_vals, s, omega_sym, j):
        """Analysiere Stabilitätsreserven für alle Systeme"""
        
        st.markdown("### 🎯 Stabilitätsreserven-Analyse")
        
        stability_data = []
        
        for tf_str, label in systems:
            try:
                G_s = safe_sympify(tf_str, {'s': s, 'omega': omega_sym})
                
                if not hasattr(G_s, 'subs'):
                    G_s = sp.sympify(G_s)
                
                G_jw = G_s.subs(s, j*omega_sym)
                
                # Numerische Auswertung für Stabilitätsanalyse
                if G_jw.is_number:
                    # Konstante haben keine Eckfrequenzen
                    stability_data.append({
                        'System': label,
                        'Eckfrequenz [rad/s]': 'N/A (Konstante)',
                        'Amplitudenreserve [dB]': 'N/A',
                        'Phasenreserve [°]': 'N/A',
                        'Stabilitätsbewertung': 'Stabil (P-Glied)'
                    })
                    continue
                
                G_func = sp.lambdify(omega_sym, G_jw, 'numpy')
                G_vals = G_func(omega_vals)
                
                # Filter infinite values
                finite_mask = np.isfinite(G_vals)
                G_clean = G_vals[finite_mask]
                omega_clean = omega_vals[finite_mask]
                
                if len(G_clean) == 0:
                    stability_data.append({
                        'System': label,
                        'Eckfrequenz [rad/s]': 'Fehler',
                        'Amplitudenreserve [dB]': 'Fehler',
                        'Phasenreserve [°]': 'Fehler',
                        'Stabilitätsbewertung': 'Analyse fehlgeschlagen'
                    })
                    continue
                
                # Berechne Magnitude und Phase
                mag_vals = np.abs(G_clean)
                phase_vals = np.angle(G_clean) * 180 / np.pi
                mag_db = 20 * np.log10(mag_vals)
                
                # Eckfrequenz (Durchtrittsfrequenz): |G(jω)| = 1 (0 dB)
                gain_crossover_freq = None
                gain_margin = None
                
                # Finde Nulldurchgang der Magnitude (0 dB)
                zero_crossings = np.where(np.diff(np.signbit(mag_db)))[0]
                if len(zero_crossings) > 0:
                    # Lineare Interpolation für genauere Eckfrequenz
                    idx = zero_crossings[0]
                    if idx < len(omega_clean) - 1:
                        gain_crossover_freq = np.interp(0, [mag_db[idx], mag_db[idx+1]], 
                                                      [omega_clean[idx], omega_clean[idx+1]])
                
                # Phasenreserve bei Eckfrequenz
                phase_margin = None
                if gain_crossover_freq is not None:
                    # Finde Phase bei Eckfrequenz
                    freq_idx = np.argmin(np.abs(omega_clean - gain_crossover_freq))
                    phase_at_crossover = phase_vals[freq_idx]
                    phase_margin = 180 + phase_at_crossover  # Phase margin
                
                # Amplitudenreserve: bei ω wo Phase = -180°
                phase_crossover_freq = None
                
                # Finde -180° Durchgang
                phase_shifted = phase_vals + 180  # Verschiebe um Nulldurchgang zu finden
                phase_crossings = np.where(np.diff(np.signbit(phase_shifted)))[0]
                
                if len(phase_crossings) > 0:
                    idx = phase_crossings[0]
                    if idx < len(omega_clean) - 1:
                        phase_crossover_freq = np.interp(0, [phase_shifted[idx], phase_shifted[idx+1]], 
                                                       [omega_clean[idx], omega_clean[idx+1]])
                        # Magnitude bei dieser Frequenz
                        freq_idx = np.argmin(np.abs(omega_clean - phase_crossover_freq))
                        gain_margin = -mag_db[freq_idx]  # Amplitudenreserve in dB
                
                # Stabilitätsbewertung
                stability_assessment = "Stabil"
                if phase_margin is not None and phase_margin < 30:
                    stability_assessment = "Grenzstabil (niedrige Phasenreserve)"
                elif gain_margin is not None and gain_margin < 6:
                    stability_assessment = "Grenzstabil (niedrige Amplitudenreserve)"
                elif phase_margin is not None and phase_margin < 0:
                    stability_assessment = "Instabil"
                elif gain_margin is not None and gain_margin < 0:
                    stability_assessment = "Instabil"
                
                stability_data.append({
                    'System': label,
                    'Eckfrequenz [rad/s]': f"{gain_crossover_freq:.3f}" if gain_crossover_freq else "N/A",
                    'Amplitudenreserve [dB]': f"{gain_margin:.1f}" if gain_margin else "N/A",
                    'Phasenreserve [°]': f"{phase_margin:.1f}" if phase_margin else "N/A",
                    'Stabilitätsbewertung': stability_assessment
                })
                
            except Exception as e:
                stability_data.append({
                    'System': label,
                    'Eckfrequenz [rad/s]': 'Fehler',
                    'Amplitudenreserve [dB]': 'Fehler',
                    'Phasenreserve [°]': 'Fehler',
                    'Stabilitätsbewertung': f'Fehler: {str(e)[:50]}...'
                })
        
        # Tabelle anzeigen
        st.dataframe(stability_data, use_container_width=True)
        
        # Zusätzliche Erklärung
        with st.expander("ℹ️ Erklärung der Stabilitätsreserven"):
            st.markdown("""
            **Eckfrequenz (Durchtrittsfrequenz):** Frequenz bei der |G(jω)| = 1 (0 dB)
            
            **Amplitudenreserve:** Wie viel die Verstärkung erhöht werden kann, bevor das System instabil wird
            - Gemessen bei der Frequenz wo die Phase -180° beträgt
            - Typische Werte: > 6 dB (gut), 3-6 dB (akzeptabel), < 3 dB (kritisch)
            
            **Phasenreserve:** Wie viel zusätzliche Phasenverzögerung toleriert werden kann
            - Gemessen bei der Eckfrequenz (0 dB)
            - Typische Werte: > 45° (gut), 30-45° (akzeptabel), < 30° (kritisch)
            
            **Stabilitätsbewertung:**
            - **Stabil:** Ausreichende Reserven vorhanden
            - **Grenzstabil:** Eine Reserve ist niedrig
            - **Instabil:** System ist instabil (negative Reserven)
            """)
    
    def _calculate_stability_margins_single(self, G_s, s, omega_sym):
        """Berechne Stabilitätsreserven für ein einzelnes System"""
        
        j = sp.I
        G_jw = G_s.subs(s, j*omega_sym)
        
        if G_jw.is_number:
            st.write("• P-Glied: Keine charakteristischen Frequenzen")
            return
        
        # Frequenzvektor für Analyse
        omega_vals = np.logspace(-2, 3, 10000)
        
        try:
            G_func = sp.lambdify(omega_sym, G_jw, 'numpy')
            G_vals = G_func(omega_vals)
            
            finite_mask = np.isfinite(G_vals)
            G_clean = G_vals[finite_mask]
            omega_clean = omega_vals[finite_mask]
            
            if len(G_clean) == 0:
                st.write("• Stabilitätsreserven: Berechnung fehlgeschlagen")
                return
            
            mag_vals = np.abs(G_clean)
            phase_vals = np.angle(G_clean) * 180 / np.pi
            mag_db = 20 * np.log10(mag_vals)
            
            # Eckfrequenz
            zero_crossings = np.where(np.diff(np.signbit(mag_db)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                if idx < len(omega_clean) - 1:
                    gain_crossover_freq = np.interp(0, [mag_db[idx], mag_db[idx+1]], 
                                                  [omega_clean[idx], omega_clean[idx+1]])
                    freq_idx = np.argmin(np.abs(omega_clean - gain_crossover_freq))
                    phase_at_crossover = phase_vals[freq_idx]
                    phase_margin = 180 + phase_at_crossover
                    
                    st.write(f"• Eckfrequenz: ωc = {gain_crossover_freq:.3f} rad/s")
                    st.write(f"• Phasenreserve: φR = {phase_margin:.1f}°")
                else:
                    st.write("• Eckfrequenz: Nicht gefunden")
            else:
                st.write("• Eckfrequenz: Nicht gefunden (System hat keinen 0dB-Durchgang)")
            
            # Amplitudenreserve
            phase_shifted = phase_vals + 180
            phase_crossings = np.where(np.diff(np.signbit(phase_shifted)))[0]
            
            if len(phase_crossings) > 0:
                idx = phase_crossings[0]
                if idx < len(omega_clean) - 1:
                    phase_crossover_freq = np.interp(0, [phase_shifted[idx], phase_shifted[idx+1]], 
                                                   [omega_clean[idx], omega_clean[idx+1]])
                    freq_idx = np.argmin(np.abs(omega_clean - phase_crossover_freq))
                    gain_margin = -mag_db[freq_idx]
                    
                    st.write(f"• Amplitudenreserve: AR = {gain_margin:.1f} dB (bei ω = {phase_crossover_freq:.3f} rad/s)")
                else:
                    st.write("• Amplitudenreserve: Nicht berechenbar")
            else:
                st.write("• Amplitudenreserve: Nicht gefunden (kein -180° Phasendurchgang)")
                
        except Exception as e:
            st.write(f"• Stabilitätsreserven: Fehler bei Berechnung - {e}")
    
    
