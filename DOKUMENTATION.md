# Regelungstechnik-Rechner - Vollständige Dokumentation

## 🎯 Überblick

Der Regelungstechnik-Rechner ist ein umfassendes Tool für die Regelungstechnik mit folgenden Hauptmodulen:

### ✅ Vollständig implementierte Module:

1. **LTI-Systeme & Zustandsraumdarstellung**
   - Differentialgleichung → Zustandsraumdarstellung 
   - Übertragungsfunktion → Zustandsraumdarstellung
   - Systemanalyse (Stabilität, Steuerbarkeit, Beobachtbarkeit)

2. **Zustandstransformation**
   - Jordan-Normalform mit kompletter Herleitung
   - Regelungsnormalform (Steuerungsnormalform)
   - Transformationsmatrizen-Berechnung

3. **Linearisierung**
   - Autonome Systeme (ẋ = f(x))
   - Systeme mit Eingang (ẋ = f(x,u))
   - Stabilitätsanalyse der linearisierten Systeme

4. **Laplace-Transformation**
   - Direkte Laplace-Transformation
   - Inverse Laplace-Transformation
   - Übertragungsfunktionsanalyse
   - Pol-Nullstellen-Diagramme
   - Sprungantwort-Berechnung

## 🚀 Features

### Benutzerfreundlichkeit:
- **Streamlit Web-Interface** - Läuft im Browser
- **Responsive Design** - Funktioniert auf Desktop und Tablet
- **Interaktive Eingabe** - Einfache Formulare und Beispiele
- **LaTeX-Darstellung** - Professionelle mathematische Notation

### Nachvollziehbare Rechenwege:
- **Schritt-für-Schritt Herleitung** - Jeder Rechenschritt wird erklärt
- **Jupyter-ähnliche Ausgabe** - Übersichtliche Darstellung
- **Zwischenergebnisse** - Alle wichtigen Schritte sichtbar
- **Verifikation** - Überprüfung der Ergebnisse

### Visualisierung:
- **Plotly-Diagramme** - Interaktive Plots
- **Pol-Nullstellen-Diagramme** - Grafische Systemdarstellung  
- **Sprungantworten** - Zeitverlaufsplots
- **Matrix-Darstellung** - Übersichtliche LaTeX-Matrizen

## 📚 Verwendung der Module

### 1. LTI-Systeme & Zustandsraumdarstellung

**Beispiel: System 2. Ordnung**
```
Differentialgleichung: ÿ + 3ẏ + 2y = u
Eingabe: a₂=1, a₁=3, a₀=2, b₁=0, b₀=1
```

**Was passiert:**
1. Normierung der Differentialgleichung
2. Definition der Zustandsvariablen (x₁=y, x₂=ẏ)
3. Aufstellung der Systemmatrizen A, B, C, D
4. Vollständige Herleitung mit Erklärungen

**Erweiterte Funktionen:**
- Übertragungsfunktion → Zustandsraum (Steuerungs-/Beobachtungsnormalform)
- Systemanalyse (Eigenwerte, Stabilität, Steuerbarkeit, Beobachtbarkeit)
- Sprungantwort-Simulation

### 2. Zustandstransformation

**Jordan-Normalform:**
```
Matrix A eingeben (z.B.):
0, 1
-2, -3
```

**Was passiert:**
1. Berechnung der Eigenwerte (charakteristisches Polynom)
2. Bestimmung der Eigenvektoren
3. Konstruktion der Jordan-Matrix J
4. Transformationsmatrix P berechnen
5. Verifikation: P⁻¹AP = J

**Regelungsnormalform:**
```
Matrix A: 0, 1 / -2, -3
Matrix B: 0 / 1
```

**Was passiert:**
1. Steuerbarkeitsmatrix Q berechnen
2. Steuerbarkeitstest (det(Q) ≠ 0)
3. Charakteristisches Polynom extrahieren
4. Regelungsnormalform konstruieren
5. Transformationsmatrix bestimmen

### 3. Linearisierung

**Autonomes System (z.B. Pendel):**
```
Systemgleichungen:
x2
-sin(x1) - 0.1*x2

Gleichgewichtspunkt: 0, 0
```

**Was passiert:**
1. Überprüfung des Gleichgewichtspunkts
2. Jacobi-Matrix ∂f/∂x berechnen
3. Auswertung am Gleichgewichtspunkt
4. Eigenwerte der linearisierten Matrix
5. Stabilitätsanalyse

**System mit Eingang:**
```
Systemgleichungen:
x2
-sin(x1) - 0.1*x2 + u1

Gleichgewichtspunkt x: 0, 0
Gleichgewichtspunkt u: 0
```

**Was passiert:**
1. A-Matrix: ∂f/∂x am Gleichgewichtspunkt
2. B-Matrix: ∂f/∂u am Gleichgewichtspunkt
3. Linearisiertes System: ẋ = Ax + Bu

### 4. Laplace-Transformation

**Direkte Transformation:**
```
Zeitfunktion: exp(-2*t)*cos(3*t)
```

**Was passiert:**
1. Anwendung der Laplace-Transformation
2. Vereinfachung des Ergebnisses
3. Konvergenzbedingungen
4. Anzeige in LaTeX-Format

**Inverse Transformation:**
```
Laplace-Funktion: (s+2)/((s+2)**2 + 9)
```

**Was passiert:**
1. Optional: Partialbruchzerlegung
2. Inverse Laplace-Transformation
3. Vereinfachung der Zeitfunktion
4. Vergleich mit Transformationstabelle

**Übertragungsfunktionen:**
```
Zähler: s + 2
Nenner: s**2 + 3*s + 2
```

**Was passiert:**
1. Pol- und Nullstellenberechnung
2. Stabilitätsanalyse
3. Pol-Nullstellen-Diagramm
4. Sprungantwort berechnen und plotten

## 🔧 Technische Details

### Architektur:
- **Modularer Aufbau** - Jedes Thema als eigenes Modul
- **BaseModule-Klasse** - Einheitliche Struktur
- **CalculationLogger** - Nachvollziehbare Rechenwege
- **Display-Utils** - Einheitliche Darstellung

### Verwendete Bibliotheken:
- **SymPy** - Symbolische Mathematik
- **NumPy/SciPy** - Numerische Berechnungen
- **Streamlit** - Web-Interface
- **Plotly** - Interaktive Visualisierung
- **Matplotlib** - Zusätzliche Plots
- **python-control** - Regelungstechnik-Tools

### Dateistruktur:
```
regelungstechnik/
├── main.py                    # Hauptanwendung
├── src/
│   ├── modules/
│   │   ├── base_module.py     # Basis-Klasse
│   │   ├── lti_systems.py     # LTI-Systeme (vollständig)
│   │   ├── state_transformation.py  # Transformationen (vollständig)
│   │   ├── linearization.py   # Linearisierung (vollständig)
│   │   └── laplace_transform.py     # Laplace (vollständig)
│   └── utils/
│       ├── display_utils.py   # Anzeige-Funktionen
│       └── calculation_logger.py    # Rechenweg-Logger
├── requirements.txt           # Dependencies
├── README.md                 # Basis-Dokumentation
├── testfälle.md             # Beispiele
└── beispiel.md              # Einfaches Beispiel
```

## 🎓 Pädagogischer Wert

### Lernunterstützung:
- **Komplette Herleitungen** - Jeder Schritt wird gezeigt
- **Erklärungen** - Warum wird was gemacht?
- **Verifikation** - Ergebnisse werden überprüft
- **Beispiele** - Vorgefertigte Testfälle

### Prüfungsvorbereitung:
- **Standardverfahren** - Alle wichtigen Methoden implementiert
- **Rechenweg-Protokolle** - Für Klausuren übbar
- **Fehlerbehandlung** - Typische Probleme werden erkannt
- **Sofortige Verifikation** - Ergebnisse können geprüft werden

## 🚀 Verwendung

```bash
# Installation
pip install -r requirements.txt

# Starten
streamlit run main.py

# Im Browser öffnen
http://localhost:8501
```

## 📈 Nächste Schritte

Mögliche Erweiterungen:
- **Regler-Design** - PID, Zustandsregelung
- **Frequenzgang-Analyse** - Bode, Nyquist, Nichols
- **Robustheit** - Sensitivitätsanalyse
- **Diskrete Systeme** - Z-Transformation
- **Nichtlineare Regelung** - Lyapunov, Sliding Mode
- **Export-Funktionen** - PDF-Reports, Code-Generation

Der Rechner ist bereits sehr umfangreich und deckt die wichtigsten Grundlagen der Regelungstechnik ab!
