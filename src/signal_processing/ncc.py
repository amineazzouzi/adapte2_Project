"""
NCC (corrélation croisée normalisée) vectorisée GPU/CPU — implémentation
unique partagée par tout le pipeline (suivi d'événements dans
oscillo_analysis.py, comparaison de fenêtres de référence entre signaux
dans oscillo_correlation.py).
"""

from src.core.gpu import cp, xp


def _xcorr_fft_batch(x_batch, y_batch, xp_mod):
    """
    Corrélation croisée 'same' pour un batch de paires (x_i, y_i) via FFT.
    x_batch, y_batch : (N, L) sur GPU ou CPU.
    Retourne corr : (N, L) équivalent à scipy.signal.correlate(x, y, mode='same')
    pour chaque paire, mais vectorisé sur tout le batch en un seul appel FFT.
    """
    N, L = x_batch.shape
    n_fft = 2 * L - 1
    # padding circulaire vers la taille FFT
    X = xp_mod.fft.rfft(x_batch, n=n_fft, axis=1)
    Y = xp_mod.fft.rfft(y_batch[:, ::-1], n=n_fft, axis=1)  # flip = corrélation
    corr_full = xp_mod.fft.irfft(X * Y, n=n_fft, axis=1)
    # extraire la zone centrale 'same' (taille L), comme scipy mode='same'
    start = (n_fft - L) // 2
    return corr_full[:, start:start + L]


def compute_max_ncc_batch_gpu(x_batch, y_batch, max_lag=500, xp_mod=None):
    """
    Version BATCH de compute_max_ncc : calcule la NCC max pour N paires
    de signaux simultanément sur GPU.
    x_batch, y_batch : arrays NumPy (N, L)
    Retourne : array NumPy (N,) des NCC max (0.0 si signal constant)
    """
    xp_mod = xp_mod or xp
    N, L = x_batch.shape

    xb = xp_mod.asarray(x_batch, dtype=xp_mod.float32)
    yb = xp_mod.asarray(y_batch, dtype=xp_mod.float32)

    x_c = xb - xb.mean(axis=1, keepdims=True)
    y_c = yb - yb.mean(axis=1, keepdims=True)
    std_x = xb.std(axis=1)
    std_y = yb.std(axis=1)

    corr = _xcorr_fft_batch(x_c, y_c, xp_mod)
    corr_norm = corr / (L * std_x[:, None] * std_y[:, None] + 1e-12)

    center = L // 2
    s = max(0, center - max_lag)
    e = min(L, center + max_lag + 1)

    max_vals = xp_mod.max(xp_mod.abs(corr_norm[:, s:e]), axis=1)

    # Cas dégénérés (signal constant) -> 0.0, comme l'original
    degenerate = (std_x == 0) | (std_y == 0)
    max_vals = xp_mod.where(degenerate, 0.0, max_vals)

    if xp_mod is cp:
        return cp.asnumpy(max_vals)
    import numpy as np
    return np.asarray(max_vals)


def compute_max_ncc_single(x, y, max_lag=500):
    """Version scalaire (compat. API originale), utilisée ponctuellement."""
    res = compute_max_ncc_batch_gpu(x[None, :], y[None, :], max_lag=max_lag)
    return float(res[0])


def ncc_pair_gpu_fast(x_gpu, y_gpu, max_lag=500):
    """
    NCC maximale pour deux signaux DÉJÀ GPU-résidents (cp.ndarray float32).
    Évite tout transfert CPU<->GPU dans le chemin chaud de track_events.
    Un seul sync GPU (float() sur le résultat final).
    """
    L = len(x_gpu)
    x_c = x_gpu - x_gpu.mean()
    y_c = y_gpu - y_gpu.mean()
    std_x = x_gpu.std()
    std_y = y_gpu.std()

    n_fft = 2 * L - 1
    X = cp.fft.rfft(x_c, n=n_fft)
    Y = cp.fft.rfft(y_c[::-1], n=n_fft)
    corr_full = cp.fft.irfft(X * Y, n=n_fft)
    start = (n_fft - L) // 2
    corr_norm = corr_full[start:start + L] / (L * std_x * std_y + 1e-12)

    center = L // 2
    s = max(0, center - max_lag)
    e = min(L, center + max_lag + 1)
    return float(cp.max(cp.abs(corr_norm[s:e])))
