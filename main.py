"""
Regelungstechnik-Rechner
========================
Ein modularer Rechner für Regelungstechnik mit nachvollziehbaren Rechenwegen.
"""

import streamlit as st
import sys
from pathlib import Path

# Füge das src-Verzeichnis zum Python-Pfad hinzu
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from src.modules.lti_systems import LTISystemsModule
from src.modules.state_transformation import StateTransformationModule
from src.modules.linearization import LinearizationModule
from src.modules.laplace_transform import LaplaceTransformModule
from src.modules.transfer_elements import TransferElementsModule
from src.modules.block_diagram import BlockDiagramModule
from src.modules.advanced_transfer_functions import AdvancedTransferFunctionModule
from utils.display_utils import display_step_by_step, display_latex
from utils.calculation_logger import CalculationLogger

def main():
    st.set_page_config(
        page_title="Regelungstechnik-Rechner",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 Regelungstechnik-Rechner")
    st.markdown("""
    Ein interaktiver Rechner für Regelungstechnik mit nachvollziehbaren Rechenwegen.
    Wählen Sie ein Thema aus der Seitenleiste aus.
    """)
    
    # Sidebar für Modulauswahl
    st.sidebar.title("Themen")
    
    modules = {
        "LTI-Systeme & Zustandsraumdarstellung": LTISystemsModule(),
        "Zustandstransformation": StateTransformationModule(),
        "Linearisierung": LinearizationModule(),
        "Laplace-Transformation": LaplaceTransformModule(),
        "Übertragungsglieder": TransferElementsModule(),
        "Blockschaltbild-Umformung": BlockDiagramModule(),
        "Erweiterte Übertragungsfunktionen": AdvancedTransferFunctionModule()
    }
    
    selected_module = st.sidebar.selectbox(
        "Wählen Sie ein Thema:",
        list(modules.keys())
    )
    
    # SymPy-Eingabehilfe in der Sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("📝 SymPy-Eingabehilfe", expanded=False):
        st.markdown("""
        **Grundfunktionen:**
        - Exponentialfunktion: `exp(-2*t)`, `exp(-a*t)`
        - Trigonometrische: `sin(3*t)`, `cos(w*t)`
        - Logarithmus: `log(x)`, `ln(x)`
        - Wurzel: `sqrt(x)`, `x**(1/2)`
        - Potenzen: `x**2`, `t**n`
        
        **Mathematische Konstanten:**
        - Euler'sche Zahl: `E` oder `exp(1)`
        - Pi: `pi`
        - Imaginäre Einheit: `I`
        - Unendlich: `oo`
        
        **Polynome & Brüche:**
        - Polynom: `s**3 + 2*s**2 + 3*s + 1`
        - Bruch: `(s+1)/(s**2+2*s+1)`
        - Faktorisiert: `(s+1)*(s+2)`
        
        **Besondere Funktionen:**
        - Sprungfunktion: `Heaviside(t)`
        - Dirac-Delta: `DiracDelta(t)`
        - Betrag: `Abs(x)`
        - Vorzeichen: `sign(x)`
        
        **Operatoren:**
        - Multiplikation: `*` (z.B. `2*t`, `a*x`)
        - Division: `/` (z.B. `1/s`, `x/y`)
        - Potenz: `**` (z.B. `s**2`, `x**(-1)`)
        
        **Symbolische Parameter:**
        - Verwenden Sie Buchstaben: `a`, `b`, `w`, `T`
        - Zeitvariable: `t`
        - Laplace-Variable: `s`
        
        **Beispiele:**
        - `exp(-2*t)*cos(3*t)` → Gedämpfte Schwingung
        - `1/(s**2 + 2*s + 1)` → PT2-Glied
        - `(s+1)/((s+2)*(s+3))` → Übertragungsfunktion
        """)
    
    # Info über Zuverlässigkeit
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 **Hinweis**: Dieser Rechner verwendet sowohl SymPy als auch 
    manuelle Berechnungen. Bei Problemen mit SymPy werden automatisch 
    zuverlässige manuelle Methoden verwendet.
    """)
    
    # Zeige das ausgewählte Modul
    if selected_module:
        module = modules[selected_module]
        st.header(selected_module)
        module.render()
    
    # Sidebar-Informationen
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Verfügbare Themen:")
    st.sidebar.markdown("""
    - LTI-Systeme & Zustandsraumdarstellung
    - Zustandstransformation (Jordan & Regelungsnormalform)
    - Linearisierung von Differentialgleichungen
    - Laplace-Transformation
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Entwickelt für die Regelungstechnik an der TU Braunschweig*")

if __name__ == "__main__":
    main()
