"""
Filtrage passe-bas des fenêtres oscillo — appliqué avant le calcul de la
similarité (NCC) pour retirer le bruit haute fréquence qui pollue la
corrélation croisée temporelle (voir signal_processing.ncc).

Filtre FFT idéal (brick-wall) : les composantes de fréquence > cutoff_hz
sont mises à zéro puis le signal est reconstruit par irfft. Le dt effectif
de chaque fenêtre est déduit de ses vrais timestamps (comme
compute_dominant_frequency_batch_gpu dans frequency.py) — pas de fréquence
d'échantillonnage supposée fixe.
"""

import numpy as np

from src.core.gpu import cp, xp
from src.signal_processing.frequency import time_seconds_from_axis


def lowpass_filter_batch(signals, time_axes, cutoff_hz, xp_mod=None):
    """
    signals    : array NumPy (N, L)
    time_axes  : array NumPy (N, L) (datetime64 ou numérique)
    cutoff_hz  : fréquence de coupure (Hz) — tout ce qui est au-dessus est retiré
    Retourne   : array NumPy (N, L) float32, signal filtré
    """
    xp_mod = xp_mod or xp
    N, L = signals.shape

    mask = np.ones((N, L // 2 + 1), dtype=bool)
    for i in range(N):
        t = time_seconds_from_axis(time_axes[i])
        duration = t[-1] - t[0]
        if duration <= 0:
            continue  # fenêtre dégénérée -> pas de filtrage (comme frequency.py)
        dt_eff = duration / (L - 1)
        freqs = np.fft.rfftfreq(L, d=dt_eff)
        mask[i] = freqs <= cutoff_hz

    sigs = xp_mod.asarray(signals, dtype=xp_mod.float32)
    mask_xp = xp_mod.asarray(mask)

    spectrum = xp_mod.fft.rfft(sigs, axis=1)
    spectrum = xp_mod.where(mask_xp, spectrum, 0)
    filtered = xp_mod.fft.irfft(spectrum, n=L, axis=1)

    if xp_mod is cp:
        return cp.asnumpy(filtered).astype(np.float32)
    return np.asarray(filtered, dtype=np.float32)
