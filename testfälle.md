# Testfälle für den Regelungstechnik-Rechner

## LTI-Systeme

### Beispiel 1: System 2. Ordnung
**Differentialgleichung:**
ÿ + 3ẏ + 2y = u

**Eingabewerte:**
- a₂ = 1, a₁ = 3, a₀ = 2
- b₁ = 0, b₀ = 1

### Beispiel 2: System 3. Ordnung  
**Differentialgleichung:**
y''' + 6ÿ + 11ẏ + 6y = u

**Eingabewerte:**
- a₃ = 1, a₂ = 6, a₁ = 11, a₀ = 6
- b₂ = 0, b₁ = 0, b₀ = 1

## Zustandstransformation

### Jordan-Normalform Beispiele:

**Beispiel 1: Einfache Eigenwerte**
```
Matrix A:
0, 1
-2, -3
```

**Beispiel 2: Mehrfache Eigenwerte**
```
Matrix A:
-1, 1
0, -1
```

**Beispiel 3: System 3. Ordnung**
```
Matrix A:
0, 1, 0
0, 0, 1
-6, -11, -6
```

### Regelungsnormalform Beispiele:

**Beispiel 1:**
```
Matrix A:
0, 1
-2, -3

Matrix B:
0
1
```

## Linearisierung

### Beispiel 1: Pendel
**Systemgleichungen:**
```
x2
-sin(x1) - 0.1*x2
```
**Gleichgewichtspunkt:** 0, 0

### Beispiel 2: Van der Pol Oszillator
**Systemgleichungen:**
```
x2
mu*(1-x1**2)*x2 - x1
```
**Gleichgewichtspunkt:** 0, 0

### Beispiel 3: System mit Eingang
**Systemgleichungen:**
```
x2
-sin(x1) - 0.1*x2 + u1
```
**Gleichgewichtspunkt x:** 0, 0
**Gleichgewichtspunkt u:** 0

## Laplace-Transformation

### Transformation Beispiele:

**Exponentialfunktion:** exp(-2*t)
**Sinusfunktion:** sin(3*t)
**Gedämpfte Schwingung:** exp(-t)*cos(2*t)
**Rampenfunktion:** t*exp(-a*t)

### Inverse Transformation Beispiele:

**Erste Ordnung:** 1/(s+2)
**Zweite Ordnung:** 4/(s**2 + 4)
**Komplexe Pole:** (s+1)/((s+1)**2 + 4)

### Übertragungsfunktionen:

**PT1-Glied:**
- Zähler: 1
- Nenner: s + 1

**PT2-Glied:**
- Zähler: 1  
- Nenner: s**2 + 2*s + 1

**Schwingfähiges Glied:**
- Zähler: 1
- Nenner: s**2 + 0.2*s + 1
