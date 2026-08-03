"""
Fréquence dominante par DFT directe sur une grille de fréquences fixe —
même méthode que le script de référence (somme de tension * exp(-j*2*pi*f*t)
sur [FMIN_HZ, FMAX_HZ] par pas de DF_HZ). Pas de FFT, pas de fenêtrage.

Vectorisé sur (fenêtre, fréquence) au lieu des boucles Python imbriquées
d'origine, et batché + envoyé sur GPU (CuPy) quand disponible — la DFT
directe reste ~5-10x plus lente qu'une FFT à volume égal (O(n_freq*L) au
lieu de O(L log L)), le batching GPU est ce qui la rend praticable en
pipeline (~190 fenêtres/s mesuré ici, contre ~7/s en boucle naïve).
"""

import numpy as np
import pandas as pd

from src.core.gpu import GPU_AVAILABLE

if GPU_AVAILABLE:
    import cupy as cp

FMIN_HZ = 10      # fréquence minimale
FMAX_HZ = 2500    # fréquence maximale
DF_HZ = 10        # résolution fréquentielle (Hz)

# Nombre de fenêtres traitées ensemble par appel DFT vectorisé. Borne la
# VRAM utilisée (tenseur de phase (batch, n_freq, L) en complex64) — au-delà
# de ~128 sur un GPU 4 Go on obtient un OutOfMemoryError ; 64 laisse de la
# marge quand plusieurs jobs du pipeline partagent le même GPU (voir
# batch_pipeline.py, un process par signal, round-robin sur les GPU).
WINDOW_BATCH_SIZE = 64


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


def _dft_amplitude_batch(xp, signals_batch, t_rel_batch, freqs, L):
    """
    DFT directe (pas de FFT), vectorisée sur un batch de fenêtres.
    signals_batch, t_rel_batch : (B, L)  ;  freqs : (n_freq,)
    Retourne (B, n_freq) — même formule que le script de référence
    (2/(L-1) * |somme signal * exp(-j*2*pi*f*t)|), juste vectorisée.
    """
    phase = xp.exp((-2j * xp.pi * freqs[None, :, None] * t_rel_batch[:, None, :]).astype(xp.complex64))
    dft = xp.einsum('bfl,bl->bf', phase, signals_batch.astype(xp.complex64))
    return xp.abs((2.0 / (L - 1)) * dft)


def _compute_amplitudes(signals, time_axes, xp_mod=None):
    """
    Calcule l'amplitude DFT (N, n_freq) pour toutes les fenêtres, batché et
    sur GPU si disponible. Retourne (amplitudes, freqs, valid_mask).
    """
    N, L = signals.shape
    freqs = np.arange(FMIN_HZ, FMAX_HZ + DF_HZ, DF_HZ).astype(np.float32)
    amplitudes = np.zeros((N, len(freqs)), dtype=np.float32)

    t_rel = np.empty((N, L), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    for i in range(N):
        t_secs = time_seconds_from_axis(time_axes[i])
        duration = t_secs[-1] - t_secs[0]
        if duration <= 0:
            continue
        valid[i] = True
        t_rel[i] = (t_secs - t_secs[0]).astype(np.float32)

    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return amplitudes, freqs, valid

    xp = xp_mod if xp_mod is not None else (cp if GPU_AVAILABLE else np)
    freqs_x = xp.asarray(freqs)

    for s in range(0, len(valid_idx), WINDOW_BATCH_SIZE):
        chunk = valid_idx[s:s + WINDOW_BATCH_SIZE]
        sb = xp.asarray(signals[chunk].astype(np.float32))
        tb = xp.asarray(t_rel[chunk])
        amp = _dft_amplitude_batch(xp, sb, tb, freqs_x, L)
        amplitudes[chunk] = cp.asnumpy(amp) if xp is not np else amp

    return amplitudes, freqs, valid


def compute_dominant_frequency_batch_gpu(signals, time_axes, n_freq,  # noqa: ARG001
                                          freq_chunk, xp_mod=None):     # noqa: ARG001
    """
    Calcule la fréquence dominante par DFT directe sur [FMIN_HZ, FMAX_HZ].

    signals    : array NumPy (N, L)
    time_axes  : array NumPy (N, L) (datetime64 ou numérique)
    Retourne   : array NumPy (N,) des fréquences dominantes (Hz)
    """
    amplitudes, freqs, valid = _compute_amplitudes(signals, time_axes, xp_mod)
    dominant_freqs = np.zeros(signals.shape[0], dtype=np.float64)
    if valid.any():
        dominant_freqs[valid] = freqs[np.argmax(amplitudes[valid], axis=1)]
    return dominant_freqs


def compute_top_n_dominant_frequencies_batch(signals, time_axes, top_n=4):
    """
    Variante de compute_dominant_frequency_batch_gpu qui retourne, pour
    chaque fenêtre, les top_n pics spectraux (amplitude décroissante) au
    lieu du seul maximum — utilisé pour l'affichage informatif sur les
    plots de référence (harmoniques / composantes secondaires du signal).

    Sélection simple par amplitude brute (top_n fréquences les plus
    fortes) : deux fréquences voisines d'un même pic peuvent donc
    apparaître ensemble dans le résultat.

    signals    : array NumPy (N, L)
    time_axes  : array NumPy (N, L) (datetime64 ou numérique)
    Retourne   : array NumPy (N, top_n) des fréquences (Hz), triées par
                 amplitude décroissante.
    """
    amplitudes, freqs, valid = _compute_amplitudes(signals, time_axes)
    N = signals.shape[0]
    top_freqs = np.zeros((N, top_n), dtype=np.float64)
    n_pick = min(top_n, len(freqs))

    for i in np.where(valid)[0]:
        top_idx = np.argpartition(amplitudes[i], -n_pick)[-n_pick:]
        top_idx = top_idx[np.argsort(amplitudes[i][top_idx])[::-1]]  # tri décroissant
        top_freqs[i, :n_pick] = freqs[top_idx]

    return top_freqs


def compute_dominant_frequency_single(signal, time_axis, n_freq, freq_chunk):
    """Version scalaire (compat. API originale)."""
    res = compute_dominant_frequency_batch_gpu(
        np.asarray(signal)[None, :], np.asarray(time_axis, dtype=object)[None, :],
        n_freq=n_freq, freq_chunk=freq_chunk
    )
    return float(res[0])
