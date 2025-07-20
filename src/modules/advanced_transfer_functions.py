"""
Erweiterte Übertragungsfunktionen (Nyquist, Bode, Pol-Nullstellen)
"""

import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import control as ct
from modules.base_module import BaseModule
from utils.display_utils import display_step_by_step, display_latex

class AdvancedTransferFunctionModule(BaseModule):
    """Modul für erweiterte Übertragungsfunktionsanalyse"""
    
    def __init__(self):
        super().__init__(
            "Erweiterte Übertragungsfunktionen",
            "Nyquist-Diagramm, Bode-Diagramm, Pol-Nullstellen-Diagramm und Stabilitätsanalyse"
        )
    
    def render(self):
        self.display_description()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Pol-Nullstellen-Diagramm",
            "Bode-Diagramm", 
            "Nyquist-Diagramm",
            "Wurzelortskurve",
            "Stabilitätsanalyse"
        ])
        
        with tab1:
            self.pole_zero_plot()
        
        with tab2:
            self.bode_plot()
        
        with tab3:
            self.nyquist_plot()
        
        with tab4:
            self.root_locus()
        
        with tab5:
            self.stability_analysis()
    
    def pole_zero_plot(self):
        """Pol-Nullstellen-Diagramm"""
        st.subheader("Pol-Nullstellen-Diagramm")
        
        st.markdown("""
        **Darstellung der Pole und Nullstellen in der komplexen s-Ebene**
        - 🔴 **Pole**: Stellen wo G(s) → ∞ (Nenner = 0)
        - 🔵 **Nullstellen**: Stellen wo G(s) = 0 (Zähler = 0)
        """)
        
        # Eingabe der Übertragungsfunktion
        col1, col2 = st.columns([1, 1])
        
        with col1:
            transfer_function = st.text_input(
                "Übertragungsfunktion G(s) =",
                value="(s+2)/((s+1)*(s^2+2*s+2))",
                help="Beispiele: 1/(s+1), (s+1)/(s^2+2*s+1), 10/((s+1)*(s+5))"
            )
            
            # Analyse-Optionen
            show_grid = st.checkbox("Koordinatengitter anzeigen", value=True)
            show_stability_region = st.checkbox("Stabilitätsbereich markieren", value=True)
            show_damping_lines = st.checkbox("Dämpfungslinien", value=False)
        
        with col2:
            if st.button("Analysieren", key="pz_analyze"):
                try:
                    # SymPy-Analyse
                    s = sp.Symbol('s')
                    G = sp.sympify(transfer_function)
                    
                    # Zähler und Nenner extrahieren
                    numerator = sp.numer(G)
                    denominator = sp.denom(G)
                    
                    # Pole und Nullstellen berechnen
                    poles = sp.solve(denominator, s)
                    zeros = sp.solve(numerator, s)
                    
                    # Informationen anzeigen
                    st.markdown("**Übertragungsfunktion:**")
                    st.latex(f"G(s) = {sp.latex(G)}")
                    
                    st.markdown("**Pole:**")
                    for i, pole in enumerate(poles):
                        pole_val = complex(pole.evalf())
                        if abs(pole_val.imag) < 1e-10:
                            st.write(f"Pol {i+1}: {pole_val.real:.3f} (reell)")
                        else:
                            st.write(f"Pol {i+1}: {pole_val.real:.3f} ± {abs(pole_val.imag):.3f}j")
                    
                    st.markdown("**Nullstellen:**")
                    if zeros:
                        for i, zero in enumerate(zeros):
                            zero_val = complex(zero.evalf())
                            if abs(zero_val.imag) < 1e-10:
                                st.write(f"Nullstelle {i+1}: {zero_val.real:.3f} (reell)")
                            else:
                                st.write(f"Nullstelle {i+1}: {zero_val.real:.3f} ± {abs(zero_val.imag):.3f}j")
                    else:
                        st.write("Keine endlichen Nullstellen")
                    
                    # Pol-Nullstellen-Diagramm erstellen
                    fig = go.Figure()
                    
                    # Stabilität sbereiche
                    if show_stability_region:
                        fig.add_shape(
                            type="rect",
                            x0=-10, x1=0, y0=-10, y1=10,
                            fillcolor="lightgreen",
                            opacity=0.2,
                            line_width=0,
                            layer="below"
                        )
                        fig.add_annotation(
                            x=-5, y=8,
                            text="Stabil",
                            showarrow=False,
                            font=dict(color="green", size=14)
                        )
                        fig.add_annotation(
                            x=2, y=8,
                            text="Instabil",
                            showarrow=False,
                            font=dict(color="red", size=14)
                        )
                    
                    # Dämpfungslinien
                    if show_damping_lines:
                        for zeta in [0.1, 0.3, 0.5, 0.7, 0.9]:
                            angle = np.arccos(zeta)
                            x_line = np.linspace(-10, 0, 100)
                            y_line_pos = -x_line * np.tan(angle)
                            y_line_neg = x_line * np.tan(angle)
                            
                            fig.add_trace(go.Scatter(
                                x=x_line, y=y_line_pos,
                                mode='lines',
                                line=dict(color='gray', dash='dash', width=1),
                                showlegend=False,
                                hovertemplate=f'ζ = {zeta}'
                            ))
                            fig.add_trace(go.Scatter(
                                x=x_line, y=y_line_neg,
                                mode='lines',
                                line=dict(color='gray', dash='dash', width=1),
                                showlegend=False,
                                hovertemplate=f'ζ = {zeta}'
                            ))
                    
                    # Pole plotten
                    pole_real = [complex(pole.evalf()).real for pole in poles]
                    pole_imag = [complex(pole.evalf()).imag for pole in poles]
                    
                    fig.add_trace(go.Scatter(
                        x=pole_real, y=pole_imag,
                        mode='markers',
                        marker=dict(symbol='x', size=12, color='red', line=dict(width=2)),
                        name='Pole',
                        hovertemplate='Pol: %{x:.3f} + %{y:.3f}j<extra></extra>'
                    ))
                    
                    # Nullstellen plotten
                    if zeros:
                        zero_real = [complex(zero.evalf()).real for zero in zeros]
                        zero_imag = [complex(zero.evalf()).imag for zero in zeros]
                        
                        fig.add_trace(go.Scatter(
                            x=zero_real, y=zero_imag,
                            mode='markers',
                            marker=dict(symbol='circle-open', size=12, color='blue', line=dict(width=2)),
                            name='Nullstellen',
                            hovertemplate='Nullstelle: %{x:.3f} + %{y:.3f}j<extra></extra>'
                        ))
                    
                    # Imaginäre Achse
                    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
                    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
                    
                    # Layout
                    fig.update_layout(
                        title="Pol-Nullstellen-Diagramm",
                        xaxis_title="Realteil σ",
                        yaxis_title="Imaginärteil jω",
                        showlegend=True,
                        width=700, height=500,
                        xaxis=dict(showgrid=show_grid),
                        yaxis=dict(showgrid=show_grid, scaleanchor="x", scaleratio=1)
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Stabilitätsanalyse
                    stable = all(complex(pole.evalf()).real < 0 for pole in poles)
                    
                    if stable:
                        st.success("✅ **System ist stabil** - Alle Pole in der linken Halbebene")
                    else:
                        st.error("❌ **System ist instabil** - Mindestens ein Pol in der rechten Halbebene")
                    
                    # Zeitkonstanten und Eigenfrequenzen
                    st.markdown("**Systemeigenschaften:**")
                    for i, pole in enumerate(poles):
                        pole_val = complex(pole.evalf())
                        if abs(pole_val.imag) < 1e-10:
                            tau = -1/pole_val.real if pole_val.real != 0 else float('inf')
                            st.write(f"Pol {i+1}: Zeitkonstante τ = {tau:.3f} s")
                        else:
                            omega_n = abs(pole_val)
                            zeta = -pole_val.real / omega_n
                            st.write(f"Pol {i+1}: ωₙ = {omega_n:.3f} rad/s, ζ = {zeta:.3f}")
                    
                except Exception as e:
                    st.error(f"Fehler bei der Analyse: {e}")
                    st.info("Tipp: Verwenden Sie * für Multiplikation und ** für Potenzen")
    
    def bode_plot(self):
        """Bode-Diagramm"""
        st.subheader("Bode-Diagramm")
        
        st.markdown("""
        **Frequenzgang-Darstellung:**
        - **Amplitudengang**: |G(jω)| in dB über log(ω)
        - **Phasengang**: ∠G(jω) in Grad über log(ω)
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            transfer_function = st.text_input(
                "Übertragungsfunktion G(s) =",
                value="10/((s+1)*(s+10))",
                key="bode_tf",
                help="Beispiele: 1/(s+1), 10*s/(s^2+s+1)"
            )
            
            # Frequenzbereich
            omega_min = st.number_input("Minimale Frequenz [rad/s]:", value=0.1, format="%.2f")
            omega_max = st.number_input("Maximale Frequenz [rad/s]:", value=100.0, format="%.1f")
            
            # Optionen
            show_asymptotes = st.checkbox("Asymptoten anzeigen", value=True)
            show_margins = st.checkbox("Stabilitätsränder anzeigen", value=True)
        
        with col2:
            if st.button("Bode-Diagramm erstellen", key="bode_plot"):
                try:
                    # SymPy für symbolische Analyse
                    s = sp.Symbol('s')
                    G = sp.sympify(transfer_function)
                    
                    st.latex(f"G(s) = {sp.latex(G)}")
                    
                    # Frequenzvektor
                    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), 1000)
                    
                    # Frequenzgang berechnen
                    s_vals = 1j * omega
                    G_func = sp.lambdify(s, G, 'numpy')
                    
                    try:
                        G_jw = G_func(s_vals)
                        
                        # Magnitude und Phase
                        magnitude_db = 20 * np.log10(np.abs(G_jw))
                        phase_deg = np.angle(G_jw) * 180 / np.pi
                        
                        # Subplots erstellen
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Amplitudengang', 'Phasengang'),
                            vertical_spacing=0.1
                        )
                        
                        # Amplitudengang
                        fig.add_trace(
                            go.Scatter(x=omega, y=magnitude_db, name='|G(jω)|'),
                            row=1, col=1
                        )
                        
                        # Phasengang
                        fig.add_trace(
                            go.Scatter(x=omega, y=phase_deg, name='∠G(jω)', line=dict(color='red')),
                            row=2, col=1
                        )
                        
                        # Stabilitätsränder
                        if show_margins:
                            # 0 dB Linie
                            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                                        annotation_text="0 dB", row=1, col=1)
                            
                            # -180° Linie
                            fig.add_hline(y=-180, line_dash="dash", line_color="gray",
                                        annotation_text="-180°", row=2, col=1)
                            
                            # Durchtrittsfrequenz finden
                            zero_crossings = np.where(np.diff(np.signbit(magnitude_db)))[0]
                            if len(zero_crossings) > 0:
                                omega_c = omega[zero_crossings[0]]
                                phase_margin = 180 + phase_deg[zero_crossings[0]]
                                
                                fig.add_vline(x=omega_c, line_dash="dot", line_color="green",
                                            annotation_text=f"ωc = {omega_c:.2f}")
                                
                                st.info(f"🎯 **Phasenrand**: {phase_margin:.1f}°")
                                
                                if phase_margin > 45:
                                    st.success("✅ Guter Phasenrand (> 45°)")
                                elif phase_margin > 0:
                                    st.warning("⚠️ Kleiner Phasenrand")
                                else:
                                    st.error("❌ Negativer Phasenrand - System instabil")
                        
                        # Layout
                        fig.update_xaxes(type="log", title_text="Frequenz ω [rad/s]")
                        fig.update_yaxes(title_text="Magnitude [dB]", row=1, col=1)
                        fig.update_yaxes(title_text="Phase [°]", row=2, col=1)
                        
                        fig.update_layout(
                            title="Bode-Diagramm",
                            height=600,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Charakteristische Frequenzen
                        st.markdown("**Charakteristische Frequenzen:**")
                        
                        # Pole und Nullstellen
                        poles = sp.solve(sp.denom(G), s)
                        zeros = sp.solve(sp.numer(G), s)
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown("**Pole (Eckfrequenzen):**")
                            for i, pole in enumerate(poles):
                                pole_val = complex(pole.evalf())
                                if abs(pole_val.imag) < 1e-10 and pole_val.real < 0:
                                    omega_pole = abs(pole_val.real)
                                    st.write(f"ω{i+1} = {omega_pole:.3f} rad/s")
                        
                        with col_b:
                            st.markdown("**Nullstellen:**")
                            for i, zero in enumerate(zeros):
                                zero_val = complex(zero.evalf())
                                if abs(zero_val.imag) < 1e-10:
                                    omega_zero = abs(zero_val.real)
                                    st.write(f"ω{i+1} = {omega_zero:.3f} rad/s")
                        
                    except Exception as calc_error:
                        st.error(f"Fehler bei der Frequenzgang-Berechnung: {calc_error}")
                        
                except Exception as e:
                    st.error(f"Fehler bei der Analyse: {e}")
    
    def nyquist_plot(self):
        """Nyquist-Diagramm"""
        st.subheader("Nyquist-Diagramm (Ortskurve)")
        
        st.markdown("""
        **Darstellung von G(jω) in der komplexen Ebene:**
        - Realteil vs. Imaginärteil
        - **Nyquist-Kriterium** für Stabilitätsanalyse
        - Kritischer Punkt: -1+0j
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            transfer_function = st.text_input(
                "Übertragungsfunktion G(s) =",
                value="1/(s*(s+1)*(s+2))",
                key="nyquist_tf",
                help="Beispiel für instabile Strecke mit Integrator"
            )
            
            # Frequenzbereich
            omega_min = st.number_input("Min. Frequenz [rad/s]:", value=0.001, format="%.3f", key="nyq_min")
            omega_max = st.number_input("Max. Frequenz [rad/s]:", value=100.0, format="%.1f", key="nyq_max")
            
            # Regler hinzufügen?
            add_controller = st.checkbox("Regler hinzufügen (K)")
            if add_controller:
                K = st.number_input("Reglerverstärkung K:", value=1.0, step=0.1)
        
        with col2:
            if st.button("Nyquist-Diagramm erstellen", key="nyquist_plot"):
                try:
                    s = sp.Symbol('s')
                    G = sp.sympify(transfer_function)
                    
                    if add_controller:
                        G = K * G
                        st.latex(f"G_0(s) = {K} \\cdot {sp.latex(sp.sympify(transfer_function))} = {sp.latex(G)}")
                    else:
                        st.latex(f"G(s) = {sp.latex(G)}")
                    
                    # Frequenzvektor (mit spezieller Behandlung um ω=0)
                    omega_pos = np.logspace(np.log10(omega_min), np.log10(omega_max), 500)
                    omega_neg = -omega_pos[::-1]
                    omega = np.concatenate([omega_neg, omega_pos])
                    
                    # Frequenzgang berechnen
                    s_vals = 1j * omega
                    G_func = sp.lambdify(s, G, 'numpy')
                    
                    # Problematische Frequenzen abfangen
                    try:
                        G_jw = G_func(s_vals)
                        
                        # Unendliche Werte entfernen
                        finite_mask = np.isfinite(G_jw)
                        G_jw_clean = G_jw[finite_mask]
                        omega_clean = omega[finite_mask]
                        
                        real_part = np.real(G_jw_clean)
                        imag_part = np.imag(G_jw_clean)
                        
                        # Nyquist-Plot
                        fig = go.Figure()
                        
                        # Ortskurve
                        fig.add_trace(go.Scatter(
                            x=real_part, y=imag_part,
                            mode='lines',
                            name='G(jω)',
                            line=dict(color='blue', width=2),
                            hovertemplate='Re: %{x:.3f}<br>Im: %{y:.3f}<extra></extra>'
                        ))
                        
                        # Kritischer Punkt -1+0j
                        fig.add_trace(go.Scatter(
                            x=[-1], y=[0],
                            mode='markers',
                            marker=dict(symbol='x', size=15, color='red', line=dict(width=3)),
                            name='Kritischer Punkt (-1+0j)',
                            hovertemplate='Kritischer Punkt<br>Re: -1<br>Im: 0<extra></extra>'
                        ))
                        
                        # Einheitskreis
                        theta = np.linspace(0, 2*np.pi, 100)
                        unit_circle_x = np.cos(theta)
                        unit_circle_y = np.sin(theta)
                        
                        fig.add_trace(go.Scatter(
                            x=unit_circle_x, y=unit_circle_y,
                            mode='lines',
                            line=dict(color='gray', dash='dash', width=1),
                            name='Einheitskreis',
                            showlegend=False
                        ))
                        
                        # Koordinatenachsen
                        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
                        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
                        
                        # Frequenzmarkierungen
                        marker_frequencies = [0.1, 1, 10]
                        for omega_marker in marker_frequencies:
                            if omega_min <= omega_marker <= omega_max:
                                try:
                                    G_marker = G_func(1j * omega_marker)
                                    if np.isfinite(G_marker):
                                        fig.add_trace(go.Scatter(
                                            x=[np.real(G_marker)], y=[np.imag(G_marker)],
                                            mode='markers+text',
                                            marker=dict(symbol='circle', size=8, color='orange'),
                                            text=f'ω={omega_marker}',
                                            textposition='top center',
                                            showlegend=False,
                                            hovertemplate=f'ω = {omega_marker} rad/s<br>Re: %{{x:.3f}}<br>Im: %{{y:.3f}}<extra></extra>'
                                        ))
                                except:
                                    pass
                        
                        # Layout
                        fig.update_layout(
                            title="Nyquist-Diagramm",
                            xaxis_title="Realteil",
                            yaxis_title="Imaginärteil",
                            showlegend=True,
                            width=700, height=600,
                            xaxis=dict(showgrid=True, scaleanchor="y", scaleratio=1),
                            yaxis=dict(showgrid=True)
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Nyquist-Stabilitätskriterium
                        st.markdown("**Nyquist-Stabilitätskriterium:**")
                        
                        # Vereinfachte Analyse (für typische Fälle)
                        # Prüfe, ob Ortskurve den Punkt -1+0j umschließt
                        min_distance_to_critical = np.min(np.abs(G_jw_clean + 1))
                        
                        st.info(f"🎯 **Minimaler Abstand zum kritischen Punkt**: {min_distance_to_critical:.3f}")
                        
                        if min_distance_to_critical > 1:
                            st.success("✅ Großer Sicherheitsabstand - System robust stabil")
                        elif min_distance_to_critical > 0.5:
                            st.success("✅ Ausreichender Sicherheitsabstand")
                        elif min_distance_to_critical > 0.1:
                            st.warning("⚠️ Kleiner Sicherheitsabstand - Vorsicht bei Parametervariationen")
                        else:
                            st.error("❌ Sehr kleiner Abstand - System möglicherweise instabil")
                        
                        # Verstärkungsrand
                        gain_margin = 1 / min_distance_to_critical if min_distance_to_critical < 1 else float('inf')
                        if gain_margin != float('inf'):
                            st.write(f"**Verstärkungsrand**: {20*np.log10(gain_margin):.1f} dB")
                        
                    except Exception as calc_error:
                        st.error(f"Fehler bei der Berechnung: {calc_error}")
                        st.info("Tipp: Bei Integratoren (s im Nenner) kann es numerische Probleme geben.")
                        
                except Exception as e:
                    st.error(f"Fehler bei der Analyse: {e}")
    
    def root_locus(self):
        """Wurzelortskurve"""
        st.subheader("Wurzelortskurve (Root Locus)")
        
        st.markdown("""
        **Darstellung der Pole des geschlossenen Kreises in Abhängigkeit von der Verstärkung K:**
        - Charakteristische Gleichung: $1 + K \\cdot G_0(s) = 0$
        - Pole wandern mit steigender Verstärkung K
        - Wichtig für Reglerentwurf
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Offene Kette
            open_loop = st.text_input(
                "Offene Kette G₀(s) =",
                value="1/(s*(s+1)*(s+2))",
                key="rlocus_ol",
                help="Strecke × Regler (ohne K)"
            )
            
            # K-Bereich
            k_min = st.number_input("Minimale Verstärkung K:", value=0.0, step=0.1)
            k_max = st.number_input("Maximale Verstärkung K:", value=10.0, step=0.1)
            
            # Spezielle K-Werte
            k_values = st.text_input(
                "Spezielle K-Werte (kommagetrennt):",
                value="0.5, 1, 2, 5",
                help="Pole für diese K-Werte werden markiert"
            )
        
        with col2:
            if st.button("Wurzelortskurve erstellen", key="rlocus_plot"):
                try:
                    s = sp.Symbol('s')
                    K = sp.Symbol('K', real=True, positive=True)
                    G0 = sp.sympify(open_loop)
                    
                    st.latex(f"G_0(s) = {sp.latex(G0)}")
                    
                    # Charakteristische Gleichung: 1 + K*G0(s) = 0
                    char_eq = 1 + K * G0
                    char_eq_expanded = sp.expand(char_eq)
                    
                    st.markdown("**Charakteristische Gleichung:**")
                    st.latex(f"1 + K \\cdot G_0(s) = {sp.latex(char_eq_expanded)} = 0")
                    
                    # K-Werte für Analyse
                    k_range = np.linspace(k_min, k_max, 100)
                    
                    # Spezielle K-Werte parsen
                    try:
                        special_k = [float(k.strip()) for k in k_values.split(',') if k.strip()]
                    except:
                        special_k = []
                    
                    # Wurzelortskurve berechnen
                    fig = go.Figure()
                    
                    all_poles = []
                    
                    for k_val in k_range:
                        # Charakteristische Gleichung für aktuelles K lösen
                        char_eq_numeric = char_eq.subs(K, k_val)
                        poles = sp.solve(char_eq_numeric, s)
                        
                        # Nur endliche, numerische Pole
                        numeric_poles = []
                        for pole in poles:
                            try:
                                pole_val = complex(pole.evalf())
                                if np.isfinite(pole_val):
                                    numeric_poles.append(pole_val)
                            except:
                                continue
                        
                        all_poles.append((k_val, numeric_poles))
                    
                    # Wurzelortskurve plotten
                    pole_paths = {}
                    
                    # Organisiere Pole in Pfade
                    if all_poles:
                        n_poles = len(all_poles[0][1])
                        
                        for pole_idx in range(n_poles):
                            real_path = []
                            imag_path = []
                            k_path = []
                            
                            for k_val, poles in all_poles:
                                if pole_idx < len(poles):
                                    real_path.append(poles[pole_idx].real)
                                    imag_path.append(poles[pole_idx].imag)
                                    k_path.append(k_val)
                            
                            if real_path:
                                fig.add_trace(go.Scatter(
                                    x=real_path, y=imag_path,
                                    mode='lines',
                                    name=f'Pol {pole_idx+1}',
                                    hovertemplate='K: %{customdata:.2f}<br>Pol: %{x:.3f} + %{y:.3f}j<extra></extra>',
                                    customdata=k_path
                                ))
                    
                    # Pole bei K=0 (offene Kette)
                    open_poles = sp.solve(sp.denom(G0), s)
                    if open_poles:
                        pole_real_open = [complex(pole.evalf()).real for pole in open_poles]
                        pole_imag_open = [complex(pole.evalf()).imag for pole in open_poles]
                        
                        fig.add_trace(go.Scatter(
                            x=pole_real_open, y=pole_imag_open,
                            mode='markers',
                            marker=dict(symbol='x', size=12, color='red', line=dict(width=2)),
                            name='Pole bei K=0',
                            hovertemplate='Pol bei K=0: %{x:.3f} + %{y:.3f}j<extra></extra>'
                        ))
                    
                    # Nullstellen (Pole bei K→∞)
                    zeros = sp.solve(sp.numer(G0), s)
                    if zeros:
                        zero_real = [complex(zero.evalf()).real for zero in zeros]
                        zero_imag = [complex(zero.evalf()).imag for zero in zeros]
                        
                        fig.add_trace(go.Scatter(
                            x=zero_real, y=zero_imag,
                            mode='markers',
                            marker=dict(symbol='circle-open', size=12, color='blue', line=dict(width=2)),
                            name='Nullstellen',
                            hovertemplate='Nullstelle: %{x:.3f} + %{y:.3f}j<extra></extra>'
                        ))
                    
                    # Spezielle K-Werte markieren
                    for k_special in special_k:
                        if k_min <= k_special <= k_max:
                            char_eq_special = char_eq.subs(K, k_special)
                            poles_special = sp.solve(char_eq_special, s)
                            
                            pole_real_special = []
                            pole_imag_special = []
                            
                            for pole in poles_special:
                                try:
                                    pole_val = complex(pole.evalf())
                                    if np.isfinite(pole_val):
                                        pole_real_special.append(pole_val.real)
                                        pole_imag_special.append(pole_val.imag)
                                except:
                                    continue
                            
                            if pole_real_special:
                                fig.add_trace(go.Scatter(
                                    x=pole_real_special, y=pole_imag_special,
                                    mode='markers',
                                    marker=dict(symbol='diamond', size=10, color='green'),
                                    name=f'K = {k_special}',
                                    hovertemplate=f'K = {k_special}<br>Pol: %{{x:.3f}} + %{{y:.3f}}j<extra></extra>'
                                ))
                    
                    # Koordinatenachsen
                    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
                    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
                    
                    # Stabilitätsbereich
                    fig.add_shape(
                        type="rect",
                        x0=-20, x1=0, y0=-20, y1=20,
                        fillcolor="lightgreen",
                        opacity=0.1,
                        line_width=0,
                        layer="below"
                    )
                    
                    # Layout
                    fig.update_layout(
                        title="Wurzelortskurve",
                        xaxis_title="Realteil σ",
                        yaxis_title="Imaginärteil jω",
                        showlegend=True,
                        width=700, height=600,
                        xaxis=dict(showgrid=True),
                        yaxis=dict(showgrid=True, scaleanchor="x", scaleratio=1)
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Stabilitätsanalyse
                    st.markdown("**Stabilitätsanalyse:**")
                    
                    # Finde kritische Verstärkung (wenn Pole auf imaginäre Achse)
                    critical_k = None
                    for k_val, poles in all_poles:
                        for pole in poles:
                            if abs(pole.real) < 0.01:  # Pole nahe imaginärer Achse
                                critical_k = k_val
                                break
                        if critical_k:
                            break
                    
                    if critical_k:
                        st.warning(f"⚠️ **Kritische Verstärkung**: K ≈ {critical_k:.2f}")
                        st.write("Bei höheren Verstärkungen wird das System instabil!")
                    else:
                        st.success("✅ Kein Stabilitätsverlust im betrachteten K-Bereich")
                    
                    # Reglerempfehlung
                    st.markdown("**Empfehlungen:**")
                    stable_k_max = critical_k * 0.6 if critical_k else k_max
                    st.write(f"- Empfohlener K-Bereich: 0 < K < {stable_k_max:.2f}")
                    st.write("- Für bessere Performance: PID-Regler verwenden")
                    st.write("- Pole sollten ausreichend weit links bleiben")
                    
                except Exception as e:
                    st.error(f"Fehler bei der Wurzelortskurve: {e}")
                    st.info("Tipp: Verwenden Sie einfache Übertragungsfunktionen für bessere Ergebnisse")
    
    def stability_analysis(self):
        """Umfassende Stabilitätsanalyse"""
        st.subheader("Stabilitätsanalyse")
        
        st.markdown("""
        **Verschiedene Stabilitätskriterien im Vergleich:**
        - Pol-Lage (direktes Kriterium)
        - Hurwitz-Kriterium
        - Routh-Schema
        - Nyquist-Kriterium
        """)
        
        # System eingeben
        system_type = st.selectbox(
            "System-Typ:",
            ["Übertragungsfunktion", "Charakteristisches Polynom"]
        )
        
        if system_type == "Übertragungsfunktion":
            transfer_function = st.text_input(
                "Übertragungsfunktion G(s) =",
                value="10/((s+1)*(s^2+2*s+2))",
                key="stability_tf"
            )
            
            feedback = st.checkbox("Geschlossener Kreis (negativer Rückkopplung)", value=True)
            
        else:
            char_poly = st.text_input(
                "Charakteristisches Polynom =",
                value="s^3 + 3*s^2 + 3*s + 1",
                key="stability_poly",
                help="Polynom = 0 für die charakteristische Gleichung"
            )
        
        if st.button("Stabilitätsanalyse durchführen", key="stability_analyze"):
            try:
                s = sp.Symbol('s')
                
                if system_type == "Übertragungsfunktion":
                    G = sp.sympify(transfer_function)
                    st.latex(f"G(s) = {sp.latex(G)}")
                    
                    if feedback:
                        # Charakteristisches Polynom des geschlossenen Kreises
                        char_poly_expr = sp.denom(G) + sp.numer(G)
                        st.markdown("**Geschlossener Kreis**: $G_w(s) = \\frac{G(s)}{1 + G(s)}$")
                    else:
                        char_poly_expr = sp.denom(G)
                        st.markdown("**Offene Kette**")
                    
                else:
                    char_poly_expr = sp.sympify(char_poly)
                
                char_poly_expanded = sp.expand(char_poly_expr)
                st.latex(f"\\text{{Charakteristisches Polynom: }} {sp.latex(char_poly_expanded)} = 0")
                
                # 1. Pol-Lage (direktes Kriterium)
                st.markdown("### 1. Pol-Lage-Analyse")
                
                poles = sp.solve(char_poly_expanded, s)
                stable = True
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Pole des Systems:**")
                    for i, pole in enumerate(poles):
                        pole_val = complex(pole.evalf())
                        real_part = pole_val.real
                        imag_part = pole_val.imag
                        
                        if abs(imag_part) < 1e-10:
                            st.write(f"Pol {i+1}: {real_part:.3f} (reell)")
                        else:
                            st.write(f"Pol {i+1}: {real_part:.3f} ± {abs(imag_part):.3f}j")
                        
                        if real_part >= 0:
                            stable = False
                            st.error(f"❌ Pol {i+1} instabil (Re ≥ 0)")
                        else:
                            st.success(f"✅ Pol {i+1} stabil (Re < 0)")
                
                with col2:
                    if stable:
                        st.success("### ✅ System ist stabil")
                        st.write("Alle Pole in der linken Halbebene")
                    else:
                        st.error("### ❌ System ist instabil")
                        st.write("Mindestens ein Pol in der rechten Halbebene")
                
                # 2. Hurwitz-Kriterium
                st.markdown("### 2. Hurwitz-Kriterium")
                
                # Koeffizienten extrahieren
                poly_coeffs = [float(char_poly_expanded.coeff(s, i)) for i in range(sp.degree(char_poly_expanded), -1, -1)]
                n = len(poly_coeffs) - 1
                
                st.write(f"**Polynom-Grad**: n = {n}")
                st.write(f"**Koeffizienten**: {poly_coeffs}")
                
                # Notwendige Bedingung: Alle Koeffizienten > 0
                necessary_condition = all(coeff > 0 for coeff in poly_coeffs)
                
                if necessary_condition:
                    st.success("✅ Notwendige Bedingung erfüllt (alle Koeffizienten > 0)")
                else:
                    st.error("❌ Notwendige Bedingung verletzt (nicht alle Koeffizienten > 0)")
                
                # Hurwitz-Matrix (vereinfacht für niedrige Grade)
                if n <= 4 and necessary_condition:
                    if n == 1:
                        st.success("✅ System 1. Ordnung: Automatisch stabil bei a₀ > 0")
                    
                    elif n == 2:
                        # a₂s² + a₁s + a₀
                        a2, a1, a0 = poly_coeffs
                        hurwitz_stable = a2 > 0 and a1 > 0 and a0 > 0
                        
                        if hurwitz_stable:
                            st.success("✅ Hurwitz-Kriterium erfüllt")
                        else:
                            st.error("❌ Hurwitz-Kriterium verletzt")
                    
                    elif n == 3:
                        # a₃s³ + a₂s² + a₁s + a₀
                        a3, a2, a1, a0 = poly_coeffs
                        H1 = a1
                        H2 = a1 * a2 - a3 * a0
                        
                        st.write(f"H₁ = a₁ = {H1}")
                        st.write(f"H₂ = a₁a₂ - a₃a₀ = {H2}")
                        
                        hurwitz_stable = H1 > 0 and H2 > 0 and a0 > 0
                        
                        if hurwitz_stable:
                            st.success("✅ Hurwitz-Kriterium erfüllt")
                        else:
                            st.error("❌ Hurwitz-Kriterium verletzt")
                
                # 3. Routh-Schema (vereinfacht)
                st.markdown("### 3. Routh-Schema")
                
                if n <= 3:
                    st.write("**Routh-Tabelle:**")
                    
                    # Erste Zeile: s^n
                    row_sn = [poly_coeffs[i] for i in range(0, len(poly_coeffs), 2)]
                    # Zweite Zeile: s^(n-1)
                    row_sn1 = [poly_coeffs[i] for i in range(1, len(poly_coeffs), 2)]
                    
                    st.write(f"s^{n}: {row_sn}")
                    st.write(f"s^{n-1}: {row_sn1}")
                    
                    # Weitere Zeilen berechnen (vereinfacht)
                    routh_stable = all(coeff > 0 for coeff in poly_coeffs)
                    
                    if routh_stable and necessary_condition:
                        st.success("✅ Routh-Kriterium erfüllt (vereinfachte Prüfung)")
                    else:
                        st.error("❌ Routh-Kriterium verletzt")
                else:
                    st.info("ℹ️ Routh-Schema für höhere Ordnungen hier nicht implementiert")
                
                # 4. Zusammenfassung
                st.markdown("### 4. Zusammenfassung")
                
                stability_methods = {
                    "Pol-Lage": stable,
                    "Hurwitz (notwendig)": necessary_condition,
                    "Koeffizienten-Test": all(coeff > 0 for coeff in poly_coeffs)
                }
                
                st.markdown("**Stabilitätskriterien im Vergleich:**")
                for method, result in stability_methods.items():
                    if result:
                        st.success(f"✅ {method}: Stabil")
                    else:
                        st.error(f"❌ {method}: Instabil")
                
                # Empfehlungen
                st.markdown("### 💡 Empfehlungen")
                
                if stable:
                    st.success("""
                    **System ist stabil!** 
                    - Alle Stabilitätskriterien erfüllt
                    - System kann in Betrieb genommen werden
                    - Für optimale Performance: Regelparameter feinabstimmen
                    """)
                else:
                    st.error("""
                    **System ist instabil!**
                    - Regelparameter müssen angepasst werden
                    - Eventuell zusätzliche Kompensation erforderlich
                    - Stabilitätsränder prüfen (Bode/Nyquist)
                    """)
                
            except Exception as e:
                st.error(f"Fehler bei der Stabilitätsanalyse: {e}")
                st.info("Tipp: Überprüfen Sie die Syntax der Eingabe")
