"""NCC (Normalized Cross-Correlation) batch GPU/CPU."""

from __future__ import annotations

import numpy as np

from ..core.gpu import GPU_AVAILABLE, xp

try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None


def _xcorr_fft_batch(
    x_batch: "np.ndarray",
    y_batch: "np.ndarray",
    xp_mod: "object",
) -> "np.ndarray":
    """Corrélation croisée 'same' vectorisée pour un batch de paires via FFT."""
    N, L = x_batch.shape
    n_fft = 2 * L - 1
    X = xp_mod.fft.rfft(x_batch, n=n_fft, axis=1)
    Y = xp_mod.fft.rfft(y_batch[:, ::-1], n=n_fft, axis=1)
    corr_full = xp_mod.fft.irfft(X * Y, n=n_fft, axis=1)
    start = (n_fft - L) // 2
    return corr_full[:, start : start + L]


def compute_ncc_batch(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    max_lag: int = 1000,
    xp_mod: object = None,
) -> np.ndarray:
    """NCC maximale pour N paires (x_i, y_i) en batch GPU si disponible.

    Retourne array NumPy (N,) des NCC max.
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
    degenerate = (std_x == 0) | (std_y == 0)
    max_vals = xp_mod.where(degenerate, 0.0, max_vals)

    if GPU_AVAILABLE and xp_mod is cp:
        return cp.asnumpy(max_vals)
    return np.asarray(max_vals)


def compute_ncc_single(x: np.ndarray, y: np.ndarray, max_lag: int = 1000) -> float:
    """NCC scalaire pour deux signaux 1-D."""
    return float(compute_ncc_batch(x[None, :], y[None, :], max_lag=max_lag)[0])


def _ncc_pair_gpu_fast(
    x_gpu: "object",
    y_gpu: "object",
    max_lag: int = 1000,
) -> float:
    """NCC pour deux signaux déjà GPU-résidents — zéro transfert PCIe."""
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
    corr_norm = corr_full[start : start + L] / (L * std_x * std_y + 1e-12)

    center = L // 2
    s = max(0, center - max_lag)
    e = min(L, center + max_lag + 1)
    return float(cp.max(cp.abs(corr_norm[s:e])))
