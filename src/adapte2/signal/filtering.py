"""Filtrage des fenêtres par amplitude locale."""

from __future__ import annotations

import numpy as np


def filter_by_amplitude(
    windows: np.ndarray,
    n_segments: int = 10,
    threshold: float = 50.0,
) -> np.ndarray:
    """Classe chaque fenêtre comme anomalie (1) ou normale (0).

    Découpe chaque signal en n_segments, calcule (max-min) par segment,
    puis leur moyenne. Si la moyenne dépasse threshold → anomalie.
    Retourne array int (N,).
    """
    N, L = windows.shape
    seg_len = L // n_segments
    trimmed = windows[:, : seg_len * n_segments].reshape(N, n_segments, seg_len)
    seg_ranges = trimmed.max(axis=2) - trimmed.min(axis=2)
    mean_range = seg_ranges.mean(axis=1)
    return (mean_range > threshold).astype(int)
