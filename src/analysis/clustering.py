"""
Clustering des événements en "types" (2e niveau, inter-événements par NCC),
et construction du SignalProfile structuré à partir des sorties du pipeline.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from src.signal_processing.ncc import compute_max_ncc_batch_gpu


def cluster_events_by_type(anomaly_events, windows,
                            gpu_batch_size=2048, ncc_max_lag=5000, ncc_type_threshold=0.4):
    """
    2e niveau de clustering : NCC entre fenêtres de référence de tous les events.
    Deux events temporellement disjoints mais de même forme → même type.
    Retourne (labels: list[int], ncc_matrix: np.ndarray (N×N)).
    """
    N = len(anomaly_events)
    if N == 0:
        return [], np.zeros((0, 0))
    if N == 1:
        return [0], np.ones((1, 1))

    ref_indices = np.array([ev['indices'][0] for ev in anomaly_events])
    ref_windows = windows[ref_indices]

    # Toutes les paires i<j en un batch chunké
    pairs_i, pairs_j = np.triu_indices(N, k=1)
    ncc_flat = np.zeros(len(pairs_i), dtype=np.float64)

    for start in range(0, len(pairs_i), gpu_batch_size):
        end = min(start + gpu_batch_size, len(pairs_i))
        ncc_flat[start:end] = compute_max_ncc_batch_gpu(
            ref_windows[pairs_i[start:end]],
            ref_windows[pairs_j[start:end]],
            max_lag=ncc_max_lag
        )

    ncc_matrix = np.eye(N, dtype=np.float64)
    ncc_matrix[pairs_i, pairs_j] = ncc_flat
    ncc_matrix[pairs_j, pairs_i] = ncc_flat

    dist_condensed = squareform(np.clip(1.0 - ncc_matrix, 0.0, 1.0), checks=False)
    Z = linkage(dist_condensed, method='average')
    raw_labels = fcluster(Z, t=1.0 - ncc_type_threshold, criterion='distance')

    zero_labels = (raw_labels - 1).tolist()

    # Renumérotation par ordre de première apparition (Type 0, 1, 2, …)
    seen: dict = {}
    next_id = 0
    remapped = []
    for lbl in zero_labels:
        if lbl not in seen:
            seen[lbl] = next_id
            next_id += 1
        remapped.append(seen[lbl])

    return remapped, ncc_matrix


def build_signal_profile(anomaly_events, windows, time_arrays,
                          dom_freq_map, ncc_map, cluster_labels,
                          dom_freqs4_map=None, signal_id="signal"):
    """
    Construit un dict structuré (SignalProfile) à partir des sorties du pipeline.
    Contient toutes les métadonnées nécessaires pour la corrélation et les exports.
    """
    dom_freqs4_map = dom_freqs4_map or {}
    events_out = []
    for ev_idx, event in enumerate(anomaly_events):
        indices   = event['indices']
        ref_idx   = indices[0]
        win_ts    = [pd.to_datetime(time_arrays[i][0]) for i in indices]
        debut_ts  = pd.to_datetime(event['debut'])
        fin_ts    = pd.to_datetime(event['fin'])
        duration  = max((fin_ts - debut_ts).total_seconds(), 0.0)

        events_out.append({
            'event_id':          ev_idx,
            'cluster_id':        cluster_labels[ev_idx],
            'ref_win_idx':       ref_idx,
            'ref_timestamp':     win_ts[0],
            'ref_window':        windows[ref_idx],
            'ref_dom_freq':      float(dom_freq_map.get(ref_idx, 0.0)),
            'ref_dom_freqs4':    [float(f) for f in dom_freqs4_map.get(ref_idx, [0.0, 0.0, 0.0, 0.0])],
            'window_timestamps': win_ts,
            'ncc_vs_ref':        [float(ncc_map.get(i, 1.0)) for i in indices],
            'dom_freqs':         [float(dom_freq_map.get(i, 0.0)) for i in indices],
            'window_count':      len(indices),
            'duration_s':        duration,
        })

    t_start = min(e['ref_timestamp'] for e in events_out) if events_out else None
    t_end   = max(e['window_timestamps'][-1] for e in events_out) if events_out else None
    n_clust = max(cluster_labels, default=-1) + 1

    return {
        'signal_id':     signal_id,
        'events':        events_out,
        'total_windows': sum(e['window_count'] for e in events_out),
        't_start':       t_start,
        't_end':         t_end,
        'n_clusters':    n_clust,
    }
