"""
Display-Utilities für den Regelungstechnik-Rechner
"""

import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional
import numpy as np

def display_latex(expression, description: str = ""):
    """Zeigt eine SymPy-Expression in LaTeX-Format an"""
    if description:
        st.markdown(f"**{description}**")
    
    if isinstance(expression, sp.Basic):
        latex_str = sp.latex(expression)
        st.latex(latex_str)
    else:
        st.latex(str(expression))

def display_step_by_step(steps: List[Dict[str, Any]], title: str = "Rechenweg"):
    """
    Zeigt einen schrittweisen Rechenweg an
    
    Args:
        steps: Liste von Schritten mit 'description', 'expression', 'explanation'
        title: Titel für den Rechenweg
    """
    st.subheader(title)
    
    for i, step in enumerate(steps, 1):
        with st.expander(f"Schritt {i}: {step.get('description', 'Berechnung')}", expanded=True):
            if 'explanation' in step:
                st.markdown(step['explanation'])
            
            if 'expression' in step:
                display_latex(step['expression'])
            
            if 'code' in step:
                with st.code(step['code']):
                    pass

def display_matrix(matrix, title: str = "Matrix"):
    """Zeigt eine Matrix in LaTeX-Format an"""
    st.markdown(f"**{title}:**")
    if isinstance(matrix, sp.Matrix):
        latex_str = sp.latex(matrix)
        st.latex(latex_str)
    else:
        # Für NumPy-Arrays
        matrix_sp = sp.Matrix(matrix)
        latex_str = sp.latex(matrix_sp)
        st.latex(latex_str)

def plot_system_response(t, y, title: str = "Systemantwort", xlabel: str = "Zeit t [s]", ylabel: str = "Ausgangsgröße y(t)"):
    """Erstellt einen Plot der Systemantwort"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='y(t)'))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_pole_zero(poles, zeros=None, title: str = "Pol-Nullstellen-Diagramm"):
    """Erstellt ein Pol-Nullstellen-Diagramm"""
    fig = go.Figure()
    
    # Pole
    if len(poles) > 0:
        poles_real = [complex(p).real for p in poles]
        poles_imag = [complex(p).imag for p in poles]
        fig.add_trace(go.Scatter(
            x=poles_real, y=poles_imag,
            mode='markers',
            marker=dict(symbol='x', size=10, color='red'),
            name='Pole'
        ))
    
    # Nullstellen
    if zeros is not None and len(zeros) > 0:
        zeros_real = [complex(z).real for z in zeros]
        zeros_imag = [complex(z).imag for z in zeros]
        fig.add_trace(go.Scatter(
            x=zeros_real, y=zeros_imag,
            mode='markers',
            marker=dict(symbol='circle', size=8, color='blue'),
            name='Nullstellen'
        ))
    
    # Einheitskreis für diskrete Systeme
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Einheitskreis',
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Realteil",
        yaxis_title="Imaginärteil",
        template='plotly_white',
        xaxis=dict(zeroline=True, zerolinecolor='black'),
        yaxis=dict(zeroline=True, zerolinecolor='black'),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def create_input_section(title: str):
    """Erstellt einen einheitlichen Eingabebereich"""
    st.subheader(title)
    return st.container()

def display_result_section(title: str = "Ergebnis"):
    """Erstellt einen einheitlichen Ergebnisbereich"""
    st.subheader(title)
    return st.container()

def display_info_box(message: str, type: str = "info"):
    """Zeigt eine Informationsbox an"""
    if type == "info":
        st.info(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)
    elif type == "success":
        st.success(message)
