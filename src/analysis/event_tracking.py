"""
Groupement temporel des fenêtres anomalies en événements (via NCC entre
fenêtres consécutives), et pré-calcul en batch des métriques (fréquence
dominante + NCC vs référence) pour tous les événements détectés.
"""

import time

import numpy as np

from src.core.gpu import GPU_AVAILABLE, cp
from src.signal_processing.ncc import (
    compute_max_ncc_single, compute_max_ncc_batch_gpu, ncc_pair_gpu_fast,
)
from src.signal_processing.frequency import (
    compute_dominant_frequency_batch_gpu, compute_top_n_dominant_frequencies_batch,
)


def track_events_temporal_gpu(windows, is_anomaly, timestamps, threshold, max_lag):
    """
    Identique en sémantique à track_events_temporal de l'original.

    Optimisation RTX 5050 : toutes les fenêtres anomalies sont pré-chargées
    en VRAM une seule fois (float32). Le « last_window » reste GPU-résident
    entre les itérations → zéro transfert PCIe dans la boucle chaude.
    La NCC scalaire est calculée par ncc_pair_gpu_fast (1 seul sync GPU/appel).

    Fallback CPU si GPU indisponible ou VRAM insuffisante (> 4 GB pour les windows).
    """
    events = []
    cur    = None

    if GPU_AVAILABLE:
        anom_idx = np.where(is_anomaly == 1)[0]
        if len(anom_idx) == 0:
            return []

        # Pré-chargement GPU : toutes les fenêtres anomalies en float32
        # Budget : N_anom * L * 4 bytes ; pour 50 000 fen. de 5000 pts = 1 GB < 4 GB
        _vram_needed = len(anom_idx) * windows.shape[1] * 4
        if _vram_needed <= 4 * 1024 ** 3:
            windows_gpu = cp.asarray(windows[anom_idx], dtype=cp.float32)
            local_map   = {int(g): k for k, g in enumerate(anom_idx)}
            last_gpu    = None

            for i in range(len(is_anomaly)):
                if is_anomaly[i] == 1:
                    w_gpu = windows_gpu[local_map[i]]
                    if cur is None:
                        last_gpu = w_gpu
                        cur = {"debut": timestamps[i], "fin": timestamps[i],
                               "indices": [i], "sims": []}
                    else:
                        sim = ncc_pair_gpu_fast(last_gpu, w_gpu, max_lag=max_lag)
                        if sim >= threshold:
                            cur["fin"] = timestamps[i]
                            cur["indices"].append(i)
                            cur["sims"].append(sim)
                            last_gpu = w_gpu
                        else:
                            events.append(cur)
                            last_gpu = w_gpu
                            cur = {"debut": timestamps[i], "fin": timestamps[i],
                                   "indices": [i], "sims": []}
                else:
                    if cur:
                        events.append(cur)
                        cur      = None
                        last_gpu = None

            if cur:
                events.append(cur)
            del windows_gpu
            cp.get_default_memory_pool().free_all_blocks()
            return events

    # ── Fallback : GPU indisponible ou VRAM insuffisante ─────────────────
    for i in range(len(is_anomaly)):
        if is_anomaly[i] == 1:
            if cur is None:
                cur = {"debut": timestamps[i], "fin": timestamps[i],
                       "indices": [i], "last_window": windows[i], "sims": []}
            else:
                sim = compute_max_ncc_single(cur["last_window"], windows[i], max_lag=max_lag)
                if sim >= threshold:
                    cur["fin"] = timestamps[i]
                    cur["indices"].append(i)
                    cur["last_window"] = windows[i]
                    cur["sims"].append(sim)
                else:
                    events.append(cur)
                    cur = {"debut": timestamps[i], "fin": timestamps[i],
                           "indices": [i], "last_window": windows[i], "sims": []}
        else:
            if cur:
                events.append(cur)
                cur = None

    if cur:
        events.append(cur)
    return events


def precompute_all_metrics_gpu(anomaly_events, windows, windows_for_ncc, time_arrays,
                                n_freq_bins, freq_chunk, ncc_max_lag):
    """
    Calcule en batch GPU :
      - la fréquence dominante de CHAQUE fenêtre impliquée dans un événement
        (sur le signal brut, `windows`)
      - les 4 fréquences dominantes (pics spectraux) de la fenêtre de
        RÉFÉRENCE de chaque événement uniquement (sur le signal brut) —
        pour l'affichage informatif des plots de référence
      - la NCC de chaque fenêtre vs la première fenêtre de son événement
        (sur le signal filtré passe-bas, `windows_for_ncc` — voir
        signal_processing.filtering.lowpass_filter_batch)
    Retourne quatre dicts {win_idx: valeur}.
    """
    all_indices = []
    ref_indices = []  # pour chaque win_idx, l'indice de la fenêtre de référence

    for event in anomaly_events:
        indices = event["indices"]
        ref_idx = indices[0]
        for win_idx in indices:
            all_indices.append(win_idx)
            ref_indices.append(ref_idx)

    all_indices = np.array(all_indices)
    ref_indices = np.array(ref_indices)

    print(f"📊 Pré-calcul GPU : {len(all_indices)} fenêtres "
          f"(fréquence dominante + NCC vs référence)...")

    t0 = time.time()

    # ── Fréquences dominantes en batch ──────────────────────────────────
    sigs = windows[all_indices]
    times = time_arrays[all_indices]
    dom_freqs = compute_dominant_frequency_batch_gpu(
        sigs, times, n_freq=n_freq_bins, freq_chunk=freq_chunk
    )

    # ── Top 4 fréquences dominantes des fenêtres de référence uniquement ────
    unique_ref_indices = np.unique(ref_indices)
    top4_freqs = compute_top_n_dominant_frequencies_batch(
        windows[unique_ref_indices], time_arrays[unique_ref_indices], top_n=4
    )
    dom_freqs4_map = dict(zip(unique_ref_indices.tolist(), top4_freqs.tolist()))

    # ── NCC vs référence en batch (sauf les fenêtres qui SONT la référence) ──
    is_ref = (all_indices == ref_indices)
    non_ref_mask = ~is_ref

    ncc_values = np.zeros(len(all_indices), dtype=np.float64)
    if non_ref_mask.any():
        x_batch = windows_for_ncc[ref_indices[non_ref_mask]]
        y_batch = windows_for_ncc[all_indices[non_ref_mask]]
        ncc_values[non_ref_mask] = compute_max_ncc_batch_gpu(x_batch, y_batch, max_lag=ncc_max_lag)

    elapsed = time.time() - t0
    print(f"✅ Pré-calcul terminé en {elapsed:.2f}s "
          f"({len(all_indices)/elapsed:.0f} fenêtres/s)")

    dom_freq_map = dict(zip(all_indices.tolist(), dom_freqs.tolist()))
    ncc_map = dict(zip(all_indices.tolist(), ncc_values.tolist()))
    is_ref_map = dict(zip(all_indices.tolist(), is_ref.tolist()))

    return dom_freq_map, ncc_map, is_ref_map, dom_freqs4_map
