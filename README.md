# Regelungstechnik-Rechner

Ein interaktiver Rechner für Regelungstechnik mit nachvollziehbaren Rechenwegen, entwickelt mit Python und Streamlit.

## Features

- **LTI-Systeme & Zustandsraumdarstellung**: Umwandlung von Differentialgleichungen in Zustandsraumdarstellung
- **Zustandstransformation**: Jordan-Normalform und Regelungsnormalform (in Entwicklung)
- **Linearisierung**: Linearisierung nichtlinearer Systeme (in Entwicklung)
- **Laplace-Transformation**: Transformation und Analysis (in Entwicklung)

## Installation

1. Klone das Repository oder lade die Dateien herunter
2. Erstelle eine virtuelle Umgebung:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # oder
   .venv\Scripts\activate  # Windows
   ```

3. Installiere die Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```

## Verwendung

Starte den Rechner mit:
```bash
streamlit run main.py
```

Der Rechner öffnet sich im Browser unter `http://localhost:8501`.

## Projektstruktur

```
regelungstechnik/
├── main.py                 # Hauptanwendung
├── requirements.txt        # Python-Abhängigkeiten
├── README.md              # Diese Datei
├── themen.md              # Liste der behandelten Themen
└── src/                   # Quellcode
    ├── modules/           # Berechnungsmodule
    │   ├── base_module.py
    │   ├── lti_systems.py
    │   ├── state_transformation.py
    │   ├── linearization.py
    │   └── laplace_transform.py
    └── utils/             # Hilfsfunktionen
        ├── display_utils.py
        └── calculation_logger.py
```

## Aktueller Status

### ✅ Implementiert
- Grundstruktur mit modularem Aufbau
- LTI-Systeme: Umwandlung Differentialgleichung → Zustandsraumdarstellung
- Nachvollziehbare Rechenwege mit LaTeX-Darstellung
- Responsive Web-Interface

### 🚧 In Entwicklung
- Zustandstransformationen (Jordan- und Regelungsnormalform)
- Linearisierung nichtlinearer Systeme
- Laplace-Transformation und inverse Transformation
- Übertragungsfunktionsanalyse
- Systemanalyse (Stabilität, Steuerbarkeit, Beobachtbarkeit)

## Verwendete Bibliotheken

- **Streamlit**: Web-Interface
- **SymPy**: Symbolische Mathematik
- **NumPy**: Numerische Berechnungen
- **Matplotlib/Plotly**: Visualisierung
- **python-control**: Regelungstechnik-spezifische Funktionen

## Entwicklung

Das Projekt ist modular aufgebaut. Neue Themen können einfach als neue Module in `src/modules/` hinzugefügt werden.

Jedes Modul erbt von `BaseModule` und implementiert die `render()`-Methode für das User Interface.
