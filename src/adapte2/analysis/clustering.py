"""Groupement NCC des fenêtres en événements et clustering inter-événements."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..core.gpu import GPU_AVAILABLE
from ..signal.ncc import compute_ncc_batch, compute_ncc_single, _ncc_pair_gpu_fast
from ..signal.features import compute_dominant_frequency

try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None


def group_events_by_ncc(
    windows: np.ndarray,
    is_anomaly: np.ndarray,
    timestamps: list,
    threshold: float = 0.30,
    max_lag: int = 1000,
) -> list[dict]:
    """Groupe les fenêtres anomalies consécutives par NCC en événements.

    La fenêtre courante est comparée à la dernière fenêtre acceptée dans
    l'événement courant. GPU-résident si disponible (zéro transfert PCIe en boucle).
    """
    events: list[dict] = []
    cur: dict | None = None

    if GPU_AVAILABLE and cp is not None:
        anom_idx = np.where(is_anomaly == 1)[0]
        if len(anom_idx) == 0:
            return []

        vram_needed = len(anom_idx) * windows.shape[1] * 4  # float32
        if vram_needed <= 4 * 1024 ** 3:
            windows_gpu = cp.asarray(windows[anom_idx], dtype=cp.float32)
            local_map = {int(g): k for k, g in enumerate(anom_idx)}
            last_gpu = None

            for i in range(len(is_anomaly)):
                if is_anomaly[i] == 1:
                    w_gpu = windows_gpu[local_map[i]]
                    if cur is None:
                        last_gpu = w_gpu
                        cur = {
                            "debut": timestamps[i],
                            "fin": timestamps[i],
                            "indices": [i],
                            "sims": [],
                        }
                    else:
                        sim = _ncc_pair_gpu_fast(last_gpu, w_gpu, max_lag=max_lag)
                        if sim >= threshold:
                            cur["fin"] = timestamps[i]
                            cur["indices"].append(i)
                            cur["sims"].append(sim)
                            last_gpu = w_gpu
                        else:
                            events.append(cur)
                            last_gpu = w_gpu
                            cur = {
                                "debut": timestamps[i],
                                "fin": timestamps[i],
                                "indices": [i],
                                "sims": [],
                            }
                else:
                    if cur is not None:
                        events.append(cur)
                        cur = None
                        last_gpu = None

            if cur is not None:
                events.append(cur)
            del windows_gpu
            cp.get_default_memory_pool().free_all_blocks()
            return events

    # Fallback CPU
    for i in range(len(is_anomaly)):
        if is_anomaly[i] == 1:
            if cur is None:
                cur = {
                    "debut": timestamps[i],
                    "fin": timestamps[i],
                    "indices": [i],
                    "last_window": windows[i],
                    "sims": [],
                }
            else:
                sim = compute_ncc_single(cur["last_window"], windows[i], max_lag=max_lag)
                if sim >= threshold:
                    cur["fin"] = timestamps[i]
                    cur["indices"].append(i)
                    cur["last_window"] = windows[i]
                    cur["sims"].append(sim)
                else:
                    events.append(cur)
                    cur = {
                        "debut": timestamps[i],
                        "fin": timestamps[i],
                        "indices": [i],
                        "last_window": windows[i],
                        "sims": [],
                    }
        else:
            if cur is not None:
                events.append(cur)
                cur = None

    if cur is not None:
        events.append(cur)
    return events


def cluster_events_by_type(
    events: list[dict],
    windows: np.ndarray,
    ncc_type_threshold: float = 0.30,
    ncc_max_lag: int = 1000,
    gpu_batch_size: int = 2048,
) -> tuple[list[int], np.ndarray]:
    """Clustering NCC entre fenêtres de référence de tous les événements.

    Retourne (labels, ncc_matrix). Labels renumérotés par ordre d'apparition.
    """
    N = len(events)
    if N == 0:
        return [], np.zeros((0, 0))
    if N == 1:
        return [0], np.ones((1, 1))

    ref_indices = np.array([ev["indices"][0] for ev in events])
    ref_windows = windows[ref_indices]

    pairs_i, pairs_j = np.triu_indices(N, k=1)
    ncc_flat = np.zeros(len(pairs_i), dtype=np.float64)

    for start in range(0, len(pairs_i), gpu_batch_size):
        end = min(start + gpu_batch_size, len(pairs_i))
        ncc_flat[start:end] = compute_ncc_batch(
            ref_windows[pairs_i[start:end]],
            ref_windows[pairs_j[start:end]],
            max_lag=ncc_max_lag,
        )

    ncc_matrix = np.eye(N, dtype=np.float64)
    ncc_matrix[pairs_i, pairs_j] = ncc_flat
    ncc_matrix[pairs_j, pairs_i] = ncc_flat

    dist_condensed = squareform(np.clip(1.0 - ncc_matrix, 0.0, 1.0), checks=False)
    Z = linkage(dist_condensed, method="average")
    raw_labels = fcluster(Z, t=1.0 - ncc_type_threshold, criterion="distance")

    # Renumérotation par ordre de première apparition
    seen: dict[int, int] = {}
    next_id = 0
    remapped: list[int] = []
    for lbl in (raw_labels - 1).tolist():
        if lbl not in seen:
            seen[lbl] = next_id
            next_id += 1
        remapped.append(seen[lbl])

    return remapped, ncc_matrix


def precompute_metrics(
    events: list[dict],
    windows: np.ndarray,
    time_arrays: np.ndarray,
    ncc_max_lag: int = 1000,
) -> tuple[dict, dict, dict]:
    """Pré-calcule fréquence dominante et NCC vs référence pour toutes les fenêtres.

    Retourne (dom_freq_map, ncc_map, is_ref_map) indexés par indice de fenêtre.
    """
    all_indices: list[int] = []
    ref_indices: list[int] = []

    for event in events:
        indices = event["indices"]
        ref_idx = indices[0]
        for win_idx in indices:
            all_indices.append(win_idx)
            ref_indices.append(ref_idx)

    all_arr = np.array(all_indices)
    ref_arr = np.array(ref_indices)

    print(f"Pré-calcul : {len(all_arr)} fenêtres (fréquence dominante + NCC vs réf)...")

    sigs = windows[all_arr]
    times = time_arrays[all_arr]
    dom_freqs = compute_dominant_frequency(sigs, times)

    is_ref = all_arr == ref_arr
    non_ref_mask = ~is_ref
    ncc_values = np.zeros(len(all_arr), dtype=np.float64)

    if non_ref_mask.any():
        ncc_values[non_ref_mask] = compute_ncc_batch(
            windows[ref_arr[non_ref_mask]],
            windows[all_arr[non_ref_mask]],
            max_lag=ncc_max_lag,
        )

    dom_freq_map = dict(zip(all_arr.tolist(), dom_freqs.tolist()))
    ncc_map = dict(zip(all_arr.tolist(), ncc_values.tolist()))
    is_ref_map = dict(zip(all_arr.tolist(), is_ref.tolist()))

    return dom_freq_map, ncc_map, is_ref_map
