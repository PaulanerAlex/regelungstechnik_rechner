"""
Übertragungsglieder
"""

import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from modules.base_module import BaseModule
from utils.display_utils import display_step_by_step, display_latex, plot_system_response

class TransferElementsModule(BaseModule):
    """Modul für Informationen über verschiedene Übertragungsglieder"""
    
    def __init__(self):
        super().__init__(
            "Übertragungsglieder",
            "Übersicht und Analyse verschiedener regelungstechnischer Übertragungsglieder"
        )
    
    def render(self):
        self.display_description()
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Grundglieder",
            "Vergleich & Analyse",
            "Parameterstudie",
            "Zeitverhalten"
        ])
        
        with tab1:
            self.basic_elements()
        
        with tab2:
            self.comparison_analysis()
        
        with tab3:
            self.parameter_study()
        
        with tab4:
            self.time_behavior()
    
    def basic_elements(self):
        """Übersicht über Grundglieder"""
        st.subheader("Regelungstechnische Grundglieder")
        
        # Auswahlbox für Übertragungsglied
        element_type = st.selectbox(
            "Wählen Sie ein Übertragungsglied:",
            [
                "P-Glied (Proportionalglied)",
                "I-Glied (Integralglied)", 
                "D-Glied (Differentialglied)",
                "PT1-Glied (Verzögerungsglied 1. Ordnung)",
                "PT2-Glied (Verzögerungsglied 2. Ordnung)",
                "DT1-Glied (Differenzierglied mit Verzögerung)",
                "IT1-Glied (Integralglied mit Verzögerung)",
                "Totzeit-Glied"
            ]
        )
        
        if element_type == "P-Glied (Proportionalglied)":
            self._show_p_element()
        elif element_type == "I-Glied (Integralglied)":
            self._show_i_element()
        elif element_type == "D-Glied (Differentialglied)":
            self._show_d_element()
        elif element_type == "PT1-Glied (Verzögerungsglied 1. Ordnung)":
            self._show_pt1_element()
        elif element_type == "PT2-Glied (Verzögerungsglied 2. Ordnung)":
            self._show_pt2_element()
        elif element_type == "DT1-Glied (Differenzierglied mit Verzögerung)":
            self._show_dt1_element()
        elif element_type == "IT1-Glied (Integralglied mit Verzögerung)":
            self._show_it1_element()
        elif element_type == "Totzeit-Glied":
            self._show_dead_time_element()
    
    def _show_p_element(self):
        """P-Glied Information"""
        st.markdown("### P-Glied (Proportionalglied)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Übertragungsfunktion:**")
            st.latex(r"G(s) = K_P")
            
            st.markdown("**Differentialgleichung:**")
            st.latex(r"y(t) = K_P \cdot u(t)")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Verstärkung: $K_P$
            - Kein Energiespeicher
            - Verzögerungslos
            - Statisches Verhalten
            """)
        
        with col2:
            # Parameter eingeben
            K_P = st.number_input("Verstärkung K_P:", value=2.0, step=0.1)
            
            # Sprungantwort berechnen und plotten
            t = np.linspace(0, 5, 1000)
            y = K_P * np.ones_like(t)  # Sprung bei t=0
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, name=f'P-Glied (K_P={K_P})'))
            fig.update_layout(
                title="Sprungantwort",
                xaxis_title="Zeit t [s]",
                yaxis_title="Ausgangsgröße y(t)"
            )
            st.plotly_chart(fig)
    
    def _show_i_element(self):
        """I-Glied Information"""
        st.markdown("### I-Glied (Integralglied)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Übertragungsfunktion:**")
            st.latex(r"G(s) = \frac{K_I}{s}")
            
            st.markdown("**Differentialgleichung:**")
            st.latex(r"\frac{dy(t)}{dt} = K_I \cdot u(t)")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Integralbeiwert: $K_I$
            - Ein Energiespeicher
            - Rampenantwort auf Sprung
            - Dynamisches Verhalten
            """)
        
        with col2:
            K_I = st.number_input("Integralbeiwert K_I:", value=1.0, step=0.1)
            
            # Sprungantwort: y(t) = K_I * t für t >= 0
            t = np.linspace(0, 5, 1000)
            y = K_I * t
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, name=f'I-Glied (K_I={K_I})'))
            fig.update_layout(
                title="Sprungantwort",
                xaxis_title="Zeit t [s]",
                yaxis_title="Ausgangsgröße y(t)"
            )
            st.plotly_chart(fig)
    
    def _show_pt1_element(self):
        """PT1-Glied Information"""
        st.markdown("### PT1-Glied (Verzögerungsglied 1. Ordnung)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Übertragungsfunktion:**")
            st.latex(r"G(s) = \frac{K}{1 + T \cdot s}")
            
            st.markdown("**Differentialgleichung:**")
            st.latex(r"T \frac{dy(t)}{dt} + y(t) = K \cdot u(t)")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Verstärkung: $K$
            - Zeitkonstante: $T$
            - Ein Energiespeicher
            - Exponentieller Verlauf
            """)
        
        with col2:
            K = st.number_input("Verstärkung K:", value=2.0, step=0.1, key="pt1_k")
            T = st.number_input("Zeitkonstante T [s]:", value=1.0, step=0.1, key="pt1_t")
            
            # Sprungantwort: y(t) = K * (1 - exp(-t/T))
            t = np.linspace(0, 5*T, 1000)
            y = K * (1 - np.exp(-t/T))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, name=f'PT1-Glied (K={K}, T={T})'))
            
            # 63% Linie einzeichnen
            fig.add_hline(y=0.63*K, line_dash="dash", 
                         annotation_text=f"63% = {0.63*K:.2f}")
            fig.add_vline(x=T, line_dash="dash",
                         annotation_text=f"T = {T}s")
            
            fig.update_layout(
                title="Sprungantwort",
                xaxis_title="Zeit t [s]",
                yaxis_title="Ausgangsgröße y(t)"
            )
            st.plotly_chart(fig)
    
    def _show_pt2_element(self):
        """PT2-Glied Information"""
        st.markdown("### PT2-Glied (Verzögerungsglied 2. Ordnung)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Übertragungsfunktion:**")
            st.latex(r"G(s) = \frac{K}{1 + 2D T s + T^2 s^2}")
            
            st.markdown("**Differentialgleichung:**")
            st.latex(r"T^2 \frac{d^2y}{dt^2} + 2DT \frac{dy}{dt} + y = K \cdot u(t)")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Verstärkung: $K$
            - Zeitkonstante: $T$
            - Dämpfung: $D$
            - Zwei Energiespeicher
            """)
        
        with col2:
            K = st.number_input("Verstärkung K:", value=1.0, step=0.1, key="pt2_k")
            T = st.number_input("Zeitkonstante T [s]:", value=1.0, step=0.1, key="pt2_t")
            D = st.number_input("Dämpfung D:", value=0.7, step=0.1, key="pt2_d")
            
            # Verhalten je nach Dämpfung
            t = np.linspace(0, 5*T, 1000)
            
            if D > 1:  # Aperiodischer Fall
                # Zwei reelle Pole
                s1 = -1/T * (D + np.sqrt(D**2 - 1))
                s2 = -1/T * (D - np.sqrt(D**2 - 1))
                A = K * s2 / (s2 - s1)
                B = K * (-s1) / (s2 - s1)
                y = K - A * np.exp(s1 * t) - B * np.exp(s2 * t)
                behavior = "Aperiodisch (D > 1)"
                
            elif D == 1:  # Aperiodischer Grenzfall
                y = K * (1 - (1 + t/T) * np.exp(-t/T))
                behavior = "Aperiodischer Grenzfall (D = 1)"
                
            else:  # Schwingend
                omega_d = (1/T) * np.sqrt(1 - D**2)
                y = K * (1 - np.exp(-D*t/T) * (np.cos(omega_d*t) + (D/np.sqrt(1-D**2))*np.sin(omega_d*t)))
                behavior = "Schwingend (D < 1)"
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, name=f'PT2-Glied ({behavior})'))
            fig.add_hline(y=K, line_dash="dash", annotation_text=f"Endwert = {K}")
            
            fig.update_layout(
                title=f"Sprungantwort - {behavior}",
                xaxis_title="Zeit t [s]",
                yaxis_title="Ausgangsgröße y(t)"
            )
            st.plotly_chart(fig)
            
            # Zusätzliche Info über Dämpfung
            if D < 1:
                st.info(f"🔄 Schwingend: Überschwingen vorhanden")
            elif D == 1:
                st.info(f"⚖️ Optimal gedämpft: Kein Überschwingen, schnellste Einstellung")
            else:
                st.info(f"🐌 Stark gedämpft: Kein Überschwingen, langsame Einstellung")
    
    def _show_d_element(self):
        """D-Glied Information"""
        st.markdown("### D-Glied (Differentialglied)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Übertragungsfunktion:**")
            st.latex(r"G(s) = K_D \cdot s")
            
            st.markdown("**Differentialgleichung:**")
            st.latex(r"y(t) = K_D \cdot \frac{du(t)}{dt}")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Differentialbeiwert: $K_D$
            - Reagiert nur auf Änderungen
            - Dirac-Impuls bei Sprung
            - Praktisch nicht realisierbar
            """)
        
        with col2:
            st.warning("⚠️ **Hinweis**: Das ideale D-Glied ist praktisch nicht realisierbar, da es auf einen Sprung mit einem unendlich hohen Dirac-Impuls reagiert.")
            
            st.markdown("**Sprungantwort:**")
            st.latex(r"y(t) = K_D \cdot \delta(t)")
            
            st.markdown("**Praktische Realisierung:** DT1-Glied")
            st.latex(r"G(s) = \frac{K_D \cdot s}{1 + T \cdot s}")
    
    def _show_dt1_element(self):
        """DT1-Glied Information"""
        st.markdown("### DT1-Glied (Differenzierglied mit Verzögerung)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Übertragungsfunktion:**")
            st.latex(r"G(s) = \frac{K_D \cdot s}{1 + T \cdot s}")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Differentialbeiwert: $K_D$
            - Zeitkonstante: $T$
            - Praktische Realisierung des D-Glieds
            - Hochpass-Verhalten
            """)
        
        with col2:
            K_D = st.number_input("Differentialbeiwert K_D:", value=1.0, step=0.1, key="dt1_kd")
            T = st.number_input("Zeitkonstante T [s]:", value=0.1, step=0.01, key="dt1_t")
            
            # Sprungantwort: y(t) = K_D * exp(-t/T)
            t = np.linspace(0, 5*T, 1000)
            y = K_D * np.exp(-t/T)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, name=f'DT1-Glied (K_D={K_D}, T={T})'))
            
            fig.update_layout(
                title="Sprungantwort",
                xaxis_title="Zeit t [s]",
                yaxis_title="Ausgangsgröße y(t)"
            )
            st.plotly_chart(fig)
    
    def _show_it1_element(self):
        """IT1-Glied Information"""
        st.markdown("### IT1-Glied (Integralglied mit Verzögerung)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Übertragungsfunktion:**")
            st.latex(r"G(s) = \frac{K_I}{s(1 + T \cdot s)}")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Integralbeiwert: $K_I$
            - Zeitkonstante: $T$
            - Zwei Energiespeicher
            - Verzögertes Integralverhalten
            """)
        
        with col2:
            K_I = st.number_input("Integralbeiwert K_I:", value=1.0, step=0.1, key="it1_ki")
            T = st.number_input("Zeitkonstante T [s]:", value=1.0, step=0.1, key="it1_t")
            
            # Sprungantwort: y(t) = K_I * (t - T + T*exp(-t/T))
            t = np.linspace(0, 5*T, 1000)
            y = K_I * (t - T + T*np.exp(-t/T))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, name=f'IT1-Glied (K_I={K_I}, T={T})'))
            
            # Asymptote einzeichnen
            y_asymptote = K_I * (t - T)
            fig.add_trace(go.Scatter(x=t, y=y_asymptote, name='Asymptote', line=dict(dash='dash')))
            
            fig.update_layout(
                title="Sprungantwort",
                xaxis_title="Zeit t [s]",
                yaxis_title="Ausgangsgröße y(t)"
            )
            st.plotly_chart(fig)
    
    def _show_dead_time_element(self):
        """Totzeit-Glied Information"""
        st.markdown("### Totzeit-Glied")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Übertragungsfunktion:**")
            st.latex(r"G(s) = e^{-T_t \cdot s}")
            
            st.markdown("**Zeitverhalten:**")
            st.latex(r"y(t) = u(t - T_t)")
            
            st.markdown("**Eigenschaften:**")
            st.markdown("""
            - Totzeit: $T_t$
            - Keine Dämpfung/Verstärkung
            - Reine Zeitverschiebung
            - Transportprozesse
            """)
        
        with col2:
            T_t = st.number_input("Totzeit T_t [s]:", value=1.0, step=0.1, key="dead_time_tt")
            
            # Sprungantwort mit Totzeit
            t = np.linspace(0, 5, 1000)
            y = np.zeros_like(t)
            y[t >= T_t] = 1.0  # Sprung nach Totzeit
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, name=f'Totzeit-Glied (T_t={T_t}s)'))
            fig.add_vline(x=T_t, line_dash="dash", annotation_text=f"T_t = {T_t}s")
            
            fig.update_layout(
                title="Sprungantwort",
                xaxis_title="Zeit t [s]",
                yaxis_title="Ausgangsgröße y(t)"
            )
            st.plotly_chart(fig)
    
    def comparison_analysis(self):
        """Vergleichende Analyse verschiedener Glieder"""
        st.subheader("Vergleich verschiedener Übertragungsglieder")
        
        # Mehrfachauswahl
        selected_elements = st.multiselect(
            "Wählen Sie Glieder zum Vergleich:",
            ["P-Glied", "PT1-Glied", "PT2-Glied (D=0.3)", "PT2-Glied (D=0.7)", "PT2-Glied (D=1.2)"],
            default=["P-Glied", "PT1-Glied", "PT2-Glied (D=0.7)"]
        )
        
        if selected_elements:
            t = np.linspace(0, 5, 1000)
            fig = go.Figure()
            
            for element in selected_elements:
                if element == "P-Glied":
                    y = np.ones_like(t)
                    fig.add_trace(go.Scatter(x=t, y=y, name="P-Glied (K=1)"))
                
                elif element == "PT1-Glied":
                    T = 1.0
                    y = 1 - np.exp(-t/T)
                    fig.add_trace(go.Scatter(x=t, y=y, name="PT1-Glied (K=1, T=1)"))
                
                elif "PT2-Glied" in element:
                    T = 1.0
                    if "D=0.3" in element:
                        D = 0.3
                    elif "D=0.7" in element:
                        D = 0.7
                    elif "D=1.2" in element:
                        D = 1.2
                    
                    if D < 1:
                        omega_d = (1/T) * np.sqrt(1 - D**2)
                        y = 1 - np.exp(-D*t/T) * (np.cos(omega_d*t) + (D/np.sqrt(1-D**2))*np.sin(omega_d*t))
                    elif D == 1:
                        y = 1 - (1 + t/T) * np.exp(-t/T)
                    else:
                        s1 = -1/T * (D + np.sqrt(D**2 - 1))
                        s2 = -1/T * (D - np.sqrt(D**2 - 1))
                        A = s2 / (s2 - s1)
                        B = (-s1) / (s2 - s1)
                        y = 1 - A * np.exp(s1 * t) - B * np.exp(s2 * t)
                    
                    fig.add_trace(go.Scatter(x=t, y=y, name=f"PT2-Glied (K=1, T=1, D={D})"))
            
            fig.update_layout(
                title="Sprungantworten im Vergleich",
                xaxis_title="Zeit t [s]",
                yaxis_title="Ausgangsgröße y(t)",
                showlegend=True
            )
            st.plotly_chart(fig)
            
            # Charakteristische Eigenschaften Tabelle
            st.subheader("Charakteristische Eigenschaften")
            
            import pandas as pd
            
            data = {
                "Übertragungsglied": ["P-Glied", "I-Glied", "PT1-Glied", "PT2-Glied", "D-Glied", "DT1-Glied"],
                "Ordnung": [0, 1, 1, 2, 0, 1],
                "Energiespeicher": [0, 1, 1, 2, 0, 1],
                "Sprungantwort": ["Sprung", "Rampe", "Exponentiell", "Schwingend/Aperiodisch", "Dirac-Impuls", "Exponentiell abklingend"],
                "Anwendung": ["Verstärker", "Integrator", "RC-Glied", "Feder-Dämpfer", "Idealer Differenzierer", "Realer Differenzierer"]
            }
            
            df = pd.DataFrame(data)
            st.table(df)
    
    def parameter_study(self):
        """Parameterstudie für ausgewähltes Glied"""
        st.subheader("Parameterstudie")
        
        element_type = st.selectbox(
            "Wählen Sie ein Glied für die Parameterstudie:",
            ["PT1-Glied", "PT2-Glied"],
            key="param_study_element"
        )
        
        if element_type == "PT1-Glied":
            st.markdown("**PT1-Glied: Einfluss der Zeitkonstante T**")
            
            # Verschiedene T-Werte
            T_values = st.multiselect(
                "Zeitkonstanten T [s]:",
                [0.5, 1.0, 2.0, 3.0, 5.0],
                default=[0.5, 1.0, 2.0]
            )
            
            if T_values:
                t = np.linspace(0, 15, 1000)
                fig = go.Figure()
                
                for T in T_values:
                    y = 1 - np.exp(-t/T)
                    fig.add_trace(go.Scatter(x=t, y=y, name=f"T = {T}s"))
                
                fig.update_layout(
                    title="Einfluss der Zeitkonstante T auf PT1-Glied",
                    xaxis_title="Zeit t [s]",
                    yaxis_title="Ausgangsgröße y(t)"
                )
                st.plotly_chart(fig)
        
        elif element_type == "PT2-Glied":
            st.markdown("**PT2-Glied: Einfluss der Dämpfung D**")
            
            D_values = st.multiselect(
                "Dämpfungswerte D:",
                [0.1, 0.3, 0.7, 1.0, 1.5],
                default=[0.3, 0.7, 1.0, 1.5]
            )
            
            if D_values:
                t = np.linspace(0, 8, 1000)
                fig = go.Figure()
                T = 1.0
                
                for D in D_values:
                    if D < 1:
                        omega_d = (1/T) * np.sqrt(1 - D**2)
                        y = 1 - np.exp(-D*t/T) * (np.cos(omega_d*t) + (D/np.sqrt(1-D**2))*np.sin(omega_d*t))
                        behavior = "schwingend"
                    elif D == 1:
                        y = 1 - (1 + t/T) * np.exp(-t/T)
                        behavior = "aperiodisch (Grenzfall)"
                    else:
                        s1 = -1/T * (D + np.sqrt(D**2 - 1))
                        s2 = -1/T * (D - np.sqrt(D**2 - 1))
                        A = s2 / (s2 - s1)
                        B = (-s1) / (s2 - s1)
                        y = 1 - A * np.exp(s1 * t) - B * np.exp(s2 * t)
                        behavior = "aperiodisch"
                    
                    fig.add_trace(go.Scatter(x=t, y=y, name=f"D = {D} ({behavior})"))
                
                fig.update_layout(
                    title="Einfluss der Dämpfung D auf PT2-Glied",
                    xaxis_title="Zeit t [s]",
                    yaxis_title="Ausgangsgröße y(t)"
                )
                st.plotly_chart(fig)
    
    def time_behavior(self):
        """Zeitverhalten verschiedener Eingangssignale"""
        st.subheader("Zeitverhalten bei verschiedenen Eingangssignalen")
        
        # Glied auswählen
        element = st.selectbox(
            "Übertragungsglied:",
            ["PT1-Glied", "PT2-Glied"],
            key="time_behavior_element"
        )
        
        # Eingangssignal auswählen
        input_signal = st.selectbox(
            "Eingangssignal:",
            ["Sprungfunktion", "Rampenfunktion", "Sinusfunktion", "Rechteckimpuls"]
        )
        
        # Parameter
        if element == "PT1-Glied":
            K = st.number_input("Verstärkung K:", value=1.0, step=0.1, key="time_k")
            T = st.number_input("Zeitkonstante T [s]:", value=1.0, step=0.1, key="time_t")
            
        elif element == "PT2-Glied":
            K = st.number_input("Verstärkung K:", value=1.0, step=0.1, key="time_k2")
            T = st.number_input("Zeitkonstante T [s]:", value=1.0, step=0.1, key="time_t2")
            D = st.number_input("Dämpfung D:", value=0.7, step=0.1, key="time_d2")
        
        # Simulation
        if st.button("Zeitverhalten simulieren"):
            t = np.linspace(0, 10, 1000)
            
            # Eingangssignal generieren
            if input_signal == "Sprungfunktion":
                u = np.ones_like(t)
            elif input_signal == "Rampenfunktion":
                u = t
            elif input_signal == "Sinusfunktion":
                freq = st.number_input("Frequenz [Hz]:", value=0.5, step=0.1)
                u = np.sin(2*np.pi*freq*t)
            elif input_signal == "Rechteckimpuls":
                width = st.number_input("Impulsbreite [s]:", value=2.0, step=0.1)
                u = np.where((t >= 1) & (t <= 1+width), 1.0, 0.0)
            
            # Ausgangssignal berechnen (vereinfacht)
            if element == "PT1-Glied":
                # Approximation durch diskrete Simulation
                dt = t[1] - t[0]
                y = np.zeros_like(t)
                for i in range(1, len(t)):
                    y[i] = y[i-1] + dt/T * (K*u[i-1] - y[i-1])
            
            elif element == "PT2-Glied":
                # Vereinfachte Approximation
                dt = t[1] - t[0]
                y = np.zeros_like(t)
                dy = np.zeros_like(t)
                
                for i in range(1, len(t)):
                    d2y = (K*u[i-1] - y[i-1] - 2*D*T*dy[i-1]) / (T**2)
                    dy[i] = dy[i-1] + dt * d2y
                    y[i] = y[i-1] + dt * dy[i]
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=u, name="Eingangssignal u(t)", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=t, y=y, name="Ausgangssignal y(t)", line=dict(color='red')))
            
            fig.update_layout(
                title=f"{element} - Antwort auf {input_signal}",
                xaxis_title="Zeit t [s]",
                yaxis_title="Signal",
                showlegend=True
            )
            st.plotly_chart(fig)
