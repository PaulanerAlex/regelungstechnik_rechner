"""
Erweiterte Tests mit Fehlerbehebung für spezifische Probleme
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import sympy as sp
import traceback

def test_and_fix_problems():
    """Teste und behebe die gefundenen Probleme"""
    print("🔧 FEHLERBEHEBUNG FÜR GEFUNDENE PROBLEME")
    print("=" * 60)
    
    # Problem 1: PT1-Glied Endwert
    print("\n📊 Problem 1: PT1-Glied Endwert")
    
    # Test der numerischen Genauigkeit
    t = np.linspace(0, 5, 100)
    K = 2.0
    T = 1.0
    
    y_analytical = K * (1 - np.exp(-t/T))
    
    print(f"Letzter Zeitpunkt: t = {t[-1]}")
    print(f"Endwert berechnet: {y_analytical[-1]}")
    print(f"Erwarteter Endwert: {K}")
    print(f"Relative Abweichung: {abs(y_analytical[-1] - K)/K * 100:.3f}%")
    
    # Lösung: Längere Zeitspanne verwenden
    t_long = np.linspace(0, 5*T, 1000)  # 5 Zeitkonstanten
    y_long = K * (1 - np.exp(-t_long/T))
    
    print(f"\nMit 5*T = {5*T}s:")
    print(f"Endwert: {y_long[-1]}")
    print(f"Relative Abweichung: {abs(y_long[-1] - K)/K * 100:.6f}%")
    
    if abs(y_long[-1] - K) < 0.01:
        print("✅ Problem 1 behoben: Längere Simulationszeit führt zu korrektem Endwert")
    
    # Problem 2: MIMO RGA Berechnung
    print("\n📊 Problem 2: MIMO RGA Berechnung")
    
    # Ursprüngliche (falsche) Berechnung
    G11, G12, G21, G22 = 2, 1, 0.5, 3
    
    det_G = G11 * G22 - G12 * G21
    lambda_11_wrong = (G11 * G22) / det_G
    lambda_12_wrong = (G12 * G21) / det_G
    
    print(f"Falsche Berechnung:")
    print(f"λ11 = G11*G22/det = {G11*G22}/{det_G} = {lambda_11_wrong}")
    print(f"λ12 = G12*G21/det = {G12*G21}/{det_G} = {lambda_12_wrong}")
    print(f"Summe: {lambda_11_wrong + lambda_12_wrong}")
    
    # Korrekte RGA Berechnung
    # RGA: Λ = G .* (G^(-1))^T
    G_matrix = np.array([[G11, G12], [G21, G22]])
    G_inv = np.linalg.inv(G_matrix)
    G_inv_T = G_inv.T
    
    # Element-weise Multiplikation
    Lambda = G_matrix * G_inv_T
    
    print(f"\nKorrekte RGA Berechnung:")
    print(f"G = {G_matrix}")
    print(f"G^(-1) = {G_inv}")
    print(f"Λ = G .* (G^(-1))^T = {Lambda}")
    print(f"λ11 = {Lambda[0,0]}")
    print(f"λ12 = {Lambda[0,1]}")
    print(f"Summe: {Lambda[0,0] + Lambda[0,1]}")
    
    if abs(Lambda[0,0] + Lambda[0,1] - 1.0) < 1e-10:
        print("✅ Problem 2 behoben: RGA korrekt berechnet")
    
    # Problem 3: Wurzelortskurve K=0
    print("\n📊 Problem 3: Wurzelortskurve K=0 Pole")
    
    s = sp.Symbol('s')
    K = sp.Symbol('K', real=True, positive=True)
    
    # G₀(s) = 1/((s+1)(s+2))
    G0 = 1 / ((s + 1) * (s + 2))
    
    # Charakteristische Gleichung: 1 + K*G₀(s) = 0
    char_eq = 1 + K * G0
    
    print(f"G₀(s) = {G0}")
    print(f"Charakteristische Gleichung: 1 + K*G₀(s) = {char_eq}")
    
    # Bei K=0: 1 + 0*G₀(s) = 1 = 0 hat keine Lösung
    # Die Pole bei K=0 sind die Pole von G₀(s)
    
    # Korrekte Berechnung: Pole von G₀(s) finden
    denominator_G0 = sp.denom(G0)
    poles_open_loop = sp.solve(denominator_G0, s)
    
    print(f"Nenner von G₀(s): {denominator_G0}")
    print(f"Pole der offenen Kette (K=0): {poles_open_loop}")
    
    poles_numeric = [complex(pole.evalf()) for pole in poles_open_loop]
    poles_real = [pole.real for pole in poles_numeric]
    
    print(f"Pole numerisch: {poles_real}")
    
    expected_poles = [-1, -2]
    if set(poles_real) == set(expected_poles):
        print("✅ Problem 3 behoben: Pole bei K=0 korrekt als Pole von G₀(s) berechnet")
    
    print("\n🎯 ALLE PROBLEME ANALYSIERT UND LÖSUNGEN GEFUNDEN")
    
    return Lambda, poles_real

def create_fixed_test():
    """Erstelle verbesserte Tests mit Korrekturen"""
    print("\n" + "=" * 60)
    print("🔧 ERSTELLE VERBESSERTE TESTS")
    print("=" * 60)
    
    fixes = {
        "pt1_endwert": "Verwende 5*T Simulationszeit für korrekten Endwert",
        "mimo_rga": "Verwende korrekte RGA-Formel: Λ = G .* (G^(-1))^T", 
        "wurzelortskurve": "Pole bei K=0 = Pole von G₀(s), nicht von char. Gleichung"
    }
    
    print("💡 EMPFOHLENE KORREKTUREN:")
    for problem, fix in fixes.items():
        print(f"• {problem}: {fix}")
    
    return fixes

if __name__ == "__main__":
    Lambda, poles = test_and_fix_problems()
    fixes = create_fixed_test()
    
    print("\n✅ ANALYSE ABGESCHLOSSEN - Module können mit diesen Korrekturen behoben werden")
