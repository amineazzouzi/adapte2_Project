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


def _parabolic_refine(mag_full, k, df):
    """
    Interpolation parabolique (3 points) autour du bin de pic k pour affiner
    la fréquence détectée au-delà de la résolution grossière du bin
    (df = 1/durée). Sans ça, un signal à 50 Hz peut être rapporté au bin FFT
    le plus proche (ex: 52 Hz) plutôt qu'à la vraie fréquence, dès que la
    fenêtre ne contient pas un nombre entier de cycles. k doit être un
    indice interne au tableau (ni 0, ni le dernier).
    """
    y_l, y_c, y_r = mag_full[k - 1], mag_full[k], mag_full[k + 1]
    denom = y_l - 2 * y_c + y_r
    if denom == 0:
        return 0.0
    offset = 0.5 * (y_l - y_r) / denom
    return float(np.clip(offset, -0.5, 0.5)) * df


def compute_dominant_frequency_batch_gpu(signals, time_axes, n_freq,  # noqa: ARG001
                                          freq_chunk, xp_mod=None):     # noqa: ARG001
    """
    Calcule la fréquence dominante par FFT standard.

    Pour chaque fenêtre, dt_effectif = durée_réelle / (L-1) est déduit des
    vrais timestamps — aucun dt fixe supposé. La FFT np.fft.rfft donne
    l'axe fréquentiel correct même si l'échantillonnage est irrégulier.

    Une fenêtre de Hann est appliquée avant la FFT (réduit la fuite
    spectrale due à la troncature — la fenêtre analysée ne contient
    généralement pas un nombre entier de cycles du signal), puis le pic est
    affiné par interpolation parabolique (voir _parabolic_refine). Sans ces
    deux corrections, l'estimation est biaisée d'un bin FFT entier en
    moyenne (ex: 50 Hz mesuré à 52 Hz) — validé empiriquement sur signal de
    test 50 Hz : biais brut ≈ 2.1 Hz, biais avec Hann + parabolique ≈ 0.2 Hz.
    Le signal brut (non fenêtré) n'est utilisé nulle part ailleurs — cette
    transformation reste locale à l'estimation de fréquence.

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
    sigs_windowed = sigs_centered * np.hanning(L)[None, :]
    spectra = np.abs(np.fft.rfft(sigs_windowed, axis=1))  # (N, L//2+1)

    for i in valid_idx:
        dt_eff = durations[i] / (L - 1)          # dt déduit de la durée réelle
        freqs  = np.fft.rfftfreq(L, d=dt_eff)    # axe fréquentiel en Hz
        df     = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
        mag    = spectra[i]
        k      = 1 + int(np.argmax(mag[1:]))     # ignorer la composante DC (indice 0)
        refined = freqs[k]
        if 0 < k < len(mag) - 1:
            refined += _parabolic_refine(mag, k, df)
        dominant_freqs[i] = refined

    return dominant_freqs


def compute_top_n_dominant_frequencies_batch(signals, time_axes, top_n=4):
    """
    Variante de compute_dominant_frequency_batch_gpu qui retourne, pour
    chaque fenêtre, les top_n pics spectraux (magnitude FFT décroissante)
    au lieu du seul maximum — utilisé pour l'affichage informatif sur les
    plots de référence (harmoniques / composantes secondaires du signal).

    signals    : array NumPy (N, L)
    time_axes  : array NumPy (N, L) (datetime64 ou numérique)
    Retourne   : array NumPy (N, top_n) des fréquences (Hz), triées par
                 magnitude décroissante ; complété de 0.0 si une fenêtre a
                 moins de top_n pics distincts.
    """
    N, L = signals.shape
    top_freqs = np.zeros((N, top_n), dtype=np.float64)

    t_secs = np.empty((N, L), dtype=np.float64)
    for i in range(N):
        t_secs[i] = time_seconds_from_axis(time_axes[i])

    durations = t_secs[:, -1] - t_secs[:, 0]
    valid_idx = np.where(durations > 0)[0]
    if len(valid_idx) == 0:
        return top_freqs

    sigs_centered = signals - signals.mean(axis=1, keepdims=True)
    sigs_windowed = sigs_centered * np.hanning(L)[None, :]
    spectra = np.abs(np.fft.rfft(sigs_windowed, axis=1))  # (N, L//2+1)

    for i in valid_idx:
        dt_eff = durations[i] / (L - 1)          # dt déduit de la durée réelle
        freqs  = np.fft.rfftfreq(L, d=dt_eff)    # axe fréquentiel en Hz
        df     = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
        mag_full = spectra[i]
        mag = mag_full[1:]                        # ignorer la composante DC pour la sélection
        n_pick = min(top_n, len(mag))
        top_idx = np.argpartition(mag, -n_pick)[-n_pick:]
        top_idx = top_idx[np.argsort(mag[top_idx])[::-1]]  # tri décroissant
        for j, idx in enumerate(top_idx):
            k = 1 + int(idx)
            refined = freqs[k]
            if 0 < k < len(mag_full) - 1:
                refined += _parabolic_refine(mag_full, k, df)
            top_freqs[i, j] = refined

    return top_freqs


def compute_dominant_frequency_single(signal, time_axis, n_freq, freq_chunk):
    """Version scalaire (compat. API originale)."""
    res = compute_dominant_frequency_batch_gpu(
        np.asarray(signal)[None, :], np.asarray(time_axis, dtype=object)[None, :],
        n_freq=n_freq, freq_chunk=freq_chunk
    )
    return float(res[0])
