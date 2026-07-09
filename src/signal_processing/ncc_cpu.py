"""
NCC CPU-only (scipy), sans dépendance GPU/CuPy — délibérément séparé de
signal_processing.ncc, qui importe src.core.gpu (donc cupy). oscillo_correlation.py
n'a besoin que de comparer des fenêtres de référence entre signaux (pas de
volume, pas de GPU nécessaire) : le garder ici évite d'imposer une
dépendance CuPy à un script qui n'en a jamais eu besoin.
"""

import numpy as np
from scipy.signal import correlate


def max_ncc_full_range(x, y):
    """
    NCC maximale sur TOUTE la plage de décalages possibles (pas bornée par
    max_lag). Utilisée par oscillo_correlation.py pour comparer des fenêtres
    de référence entre signaux différents, où l'alignement temporel n'a pas
    de sens a priori (contrairement au suivi d'événements consécutifs du
    même signal dans oscillo_analysis.py, qui utilise signal_processing.ncc,
    bornée par max_lag). Gardée séparée : les deux usages ne sont pas
    prouvés numériquement interchangeables.
    """
    xc = x - x.mean()
    yc = y - y.mean()
    sx, sy = xc.std(), yc.std()
    if sx == 0 or sy == 0:
        return 0.0
    corr = correlate(xc, yc, mode='full')
    return float(np.max(np.abs(corr)) / (len(x) * sx * sy))
