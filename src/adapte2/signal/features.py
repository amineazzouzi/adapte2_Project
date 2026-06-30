"""Calcul des features par fenêtre de signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis


def _time_seconds(time_axis: np.ndarray) -> np.ndarray:
    """Convertit un axe temporel (datetime64 ou numérique) en secondes depuis t0."""
    try:
        t = np.asarray(time_axis, dtype="datetime64[ns]")
        secs = (t - t[0]) / np.timedelta64(1, "s")
        return secs.astype(np.float64)
    except (ValueError, TypeError):
        t = np.asarray(time_axis, dtype=float)
        return t - t[0]


def compute_dominant_frequency(
    signals: np.ndarray,
    time_axes: np.ndarray,
) -> np.ndarray:
    """Fréquence dominante par FFT pour un batch de fenêtres.

    Retourne array (N,) en Hz.
    """
    N, L = signals.shape
    dominant_freqs = np.zeros(N, dtype=np.float64)

    t_secs = np.empty((N, L), dtype=np.float64)
    for i in range(N):
        t_secs[i] = _time_seconds(time_axes[i])

    durations = t_secs[:, -1] - t_secs[:, 0]
    valid_idx = np.where(durations > 0)[0]
    if len(valid_idx) == 0:
        return dominant_freqs

    sigs_centered = signals - signals.mean(axis=1, keepdims=True)
    spectra = np.abs(np.fft.rfft(sigs_centered, axis=1))  # (N, L//2+1)

    for i in valid_idx:
        dt_eff = durations[i] / (L - 1)
        freqs = np.fft.rfftfreq(L, d=dt_eff)
        dominant_freqs[i] = freqs[1 + int(np.argmax(spectra[i, 1:]))]

    return dominant_freqs


def compute_kurtosis(signals: np.ndarray) -> np.ndarray:
    """Kurtosis (excès) par fenêtre. Retourne array (N,)."""
    return scipy_kurtosis(signals, axis=1, fisher=True, bias=True)


def compute_crest_factor(signals: np.ndarray) -> np.ndarray:
    """Facteur de crête = |max| / RMS par fenêtre. Retourne array (N,)."""
    rms = np.sqrt(np.mean(signals ** 2, axis=1))
    peak = np.max(np.abs(signals), axis=1)
    return np.where(rms > 0, peak / rms, 0.0)


def compute_spectral_entropy(signals: np.ndarray) -> np.ndarray:
    """Entropie spectrale normalisée [0, 1] par fenêtre.

    < 0.7 = signal structuré (fréquences dominantes), > 0.7 = bruit.
    Retourne array (N,).
    """
    spectra = np.abs(np.fft.rfft(signals, axis=1)) ** 2  # puissance spectrale
    total = spectra.sum(axis=1, keepdims=True)
    # Éviter division par zéro
    p = np.where(total > 0, spectra / total, 0.0)
    # Éviter log(0)
    log_p = np.where(p > 0, np.log2(p), 0.0)
    raw_entropy = -np.sum(p * log_p, axis=1)
    max_entropy = np.log2(max(spectra.shape[1], 2))
    return raw_entropy / max_entropy


def compute_feature_matrix(
    signals: np.ndarray,
    time_axes: np.ndarray,
) -> pd.DataFrame:
    """Calcule toutes les features pour un batch de fenêtres.

    Retourne DataFrame avec colonnes : dom_freq, kurtosis, crest_factor,
    spectral_entropy, rms, amplitude_range.
    """
    rms = np.sqrt(np.mean(signals ** 2, axis=1))
    amp_range = signals.max(axis=1) - signals.min(axis=1)

    return pd.DataFrame(
        {
            "dom_freq": compute_dominant_frequency(signals, time_axes),
            "kurtosis": compute_kurtosis(signals),
            "crest_factor": compute_crest_factor(signals),
            "spectral_entropy": compute_spectral_entropy(signals),
            "rms": rms,
            "amplitude_range": amp_range,
        }
    )
