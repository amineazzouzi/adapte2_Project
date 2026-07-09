"""
Fréquence dominante par FFT — calcul vectorisé (remplace une boucle Python
O(n_freq * n_pts) par une FFT standard par fenêtre).
"""

import numpy as np
import pandas as pd


def time_seconds_from_axis(time_axis):
    """Convertit un axe temporel (datetime ou numérique) en secondes depuis t0."""
    if isinstance(time_axis, pd.Series):
        t = (time_axis - time_axis.iloc[0]).dt.total_seconds().values
    else:
        try:
            time_dt = np.asarray(time_axis, dtype='datetime64[ns]')
            t = (time_dt - time_dt[0]) / np.timedelta64(1, 's')
        except ValueError:
            t = np.asarray(time_axis, dtype=float)
            t = t - t[0]
    return t.astype(np.float64)


def compute_dominant_frequency_batch_gpu(signals, time_axes, n_freq,  # noqa: ARG001
                                          freq_chunk, xp_mod=None):     # noqa: ARG001
    """
    Calcule la fréquence dominante par FFT standard.

    Pour chaque fenêtre, dt_effectif = durée_réelle / (L-1) est déduit des
    vrais timestamps — aucun dt fixe supposé. La FFT np.fft.rfft donne
    l'axe fréquentiel correct même si l'échantillonnage est irrégulier.

    signals    : array NumPy (N, L)
    time_axes  : array NumPy (N, L) (datetime64 ou numérique)
    Retourne   : array NumPy (N,) des fréquences dominantes (Hz)
    """
    N, L = signals.shape
    dominant_freqs = np.zeros(N, dtype=np.float64)

    t_secs = np.empty((N, L), dtype=np.float64)
    for i in range(N):
        t_secs[i] = time_seconds_from_axis(time_axes[i])

    durations = t_secs[:, -1] - t_secs[:, 0]
    valid_idx = np.where(durations > 0)[0]
    if len(valid_idx) == 0:
        return dominant_freqs

    # FFT vectorisée sur toutes les fenêtres en une seule opération
    sigs_centered = signals - signals.mean(axis=1, keepdims=True)
    spectra = np.abs(np.fft.rfft(sigs_centered, axis=1))  # (N, L//2+1)

    for i in valid_idx:
        dt_eff = durations[i] / (L - 1)          # dt déduit de la durée réelle
        freqs  = np.fft.rfftfreq(L, d=dt_eff)    # axe fréquentiel en Hz
        # Ignorer la composante DC (indice 0)
        dominant_freqs[i] = freqs[1 + int(np.argmax(spectra[i, 1:]))]

    return dominant_freqs


def compute_dominant_frequency_single(signal, time_axis, n_freq, freq_chunk):
    """Version scalaire (compat. API originale)."""
    res = compute_dominant_frequency_batch_gpu(
        np.asarray(signal)[None, :], np.asarray(time_axis, dtype=object)[None, :],
        n_freq=n_freq, freq_chunk=freq_chunk
    )
    return float(res[0])
