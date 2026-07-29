"""Orchestrateur du pipeline d'analyse oscillo (voir oscillo_analysis.py)."""

import os
import shutil
import time

import numpy as np
import pandas as pd

from src.core.gpu import GPU_AVAILABLE
from src.core.paths import output_dir_for, signal_id_for
from src.io.datalake_reader import load_from_datalake
from src.io.raw_file_loader import load_all_oscillo_files
from src.io.profile_io import save_type_windows, serialize_signal_profile
from src.signal_processing.windowing import filter_anomaly_windows, filter_by_peak_threshold
from src.signal_processing.filtering import lowpass_filter_batch
from src.analysis.event_tracking import track_events_temporal_gpu, precompute_all_metrics_gpu
from src.analysis.clustering import cluster_events_by_type, build_signal_profile
from src.analysis.type_registry import assign_global_types
from src.reporting.plots import plot_reference_windows, plot_all_type_summaries
from src.reporting.html_report import export_enriched_html, export_type_summary_html


class OscilloPipeline:
    """
    Charge les fenêtres oscillo (data lake ou fichiers bruts), détecte les
    anomalies, les groupe en événements (NCC temporelle), clusterise les
    événements en types (NCC inter-événements), puis exporte le rapport
    enrichi + signal_profile.json + fenêtres de référence.
    """

    def __init__(self, config):
        self.config = config

    def run(self):
        c = self.config
        t_start_total = time.time()

        # ── Résolution du dossier de sortie ─────────────────────────────
        # Un seul dossier par signal (boitier+voie), quel que soit le nombre
        # de jours combinés — pas un dossier par jour. Un nouveau run pour le
        # même signal remplace le précédent (voir SIGNALS_TO_CORRELATE dans
        # oscillo_correlation.py, qui référence aussi les profils par signal).
        if c.use_datalake:
            output_dir = output_dir_for(c.boitier, c.voie)
            signal_id  = signal_id_for(c.boitier, c.voie, c.dates)
        else:
            output_dir = c.output_dir_raw
            signal_id  = os.path.basename(os.path.normpath(c.data_path))

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("PIPELINE GPU — Groupement de Fenêtres d'Anomalies")
        print(f"   Source  : {'Data Lake Parquet' if c.use_datalake else 'Fichiers bruts'}")
        if c.use_datalake:
            print(f"   Signal  : {signal_id}")
        print(f"   Device  : {'GPU (CuPy)' if GPU_AVAILABLE else 'CPU (NumPy fallback)'}")
        print(f"   Sortie  : {os.path.abspath(output_dir)}")
        print("=" * 70 + "\n")

        # ── Chargement ────────────────────────────────────────────────
        t0 = time.time()
        if c.use_datalake:
            windows, time_arrays = load_from_datalake(
                c.data_lake_path, c.boitier, c.voie, c.dates
            )
        else:
            windows, time_arrays = load_all_oscillo_files(
                c.data_path, c.channel_index, c.num_workers
            )
        print(f"⏱️  Chargement : {time.time()-t0:.2f}s")

        if len(windows) > 0:
            timestamps = [ta[0] for ta in time_arrays]
            print(f"Timestamps OK — exemple : {pd.Timestamp(timestamps[0])}")
        else:
            timestamps = []
            print("AVERTISSEMENT : aucune fenêtre chargée.")

        # ── Filtrage des anomalies par amplitude locale ─────────────────
        if len(windows) > 0:
            t0 = time.time()
            is_anomaly = filter_anomaly_windows(
                windows, n_segments=c.anomaly_n_segments, threshold=c.anomaly_threshold
            )
            is_anomaly &= filter_by_peak_threshold(windows, threshold=c.peak_threshold)
            n_anom = int(np.sum(is_anomaly))
            print(f"Fenêtres anomalies détectées : {n_anom} / {len(is_anomaly)} "
                  f"(seuil amplitude moy. > {c.anomaly_threshold} sur {c.anomaly_n_segments} segments, "
                  f"et pic max >= {c.peak_threshold} et pic min <= -{c.peak_threshold})")
            print(f"⏱️  Filtrage : {time.time()-t0:.2f}s")
        else:
            is_anomaly = np.array([], dtype=int)

        # ── Filtrage passe-bas (avant tout calcul de similarité NCC) ─────
        # Retire le bruit haute fréquence (> lowpass_cutoff_hz) qui pollue la
        # corrélation croisée. Le signal brut reste utilisé pour la détection
        # d'anomalie par amplitude, la fréquence dominante et l'affichage.
        if len(windows) > 0:
            t0 = time.time()
            windows_ncc = lowpass_filter_batch(windows, time_arrays, cutoff_hz=c.lowpass_cutoff_hz)
            print(f"⏱️  Filtrage passe-bas ({c.lowpass_cutoff_hz:.0f} Hz) : {time.time()-t0:.2f}s")
        else:
            windows_ncc = windows

        # ── Groupement temporel par NCC (GPU) ───────────────────────────
        t0 = time.time()
        if len(is_anomaly) > 0:
            anomaly_events = track_events_temporal_gpu(
                windows_ncc, is_anomaly, timestamps,
                threshold=c.ncc_threshold, max_lag=c.ncc_max_lag
            )
            # Fenêtre de référence de chaque événement = celle qui a le plus
            # grand max (signal brut) parmi toutes les fenêtres de l'événement
            # — pas forcément la première chronologiquement.
            for event in anomaly_events:
                event["ref_idx"] = max(event["indices"], key=lambda i: windows[i].max())
            print(f"\nTotal groupes (événements) : {len(anomaly_events)}")
        else:
            anomaly_events = []
            print("Aucune fenêtre à traiter.")
        print(f"⏱️  Groupement NCC : {time.time()-t0:.2f}s")

        # ── Pré-calcul GPU des métriques ────────────────────────────────
        dom_freq_map, ncc_map, dom_freqs4_map = {}, {}, {}
        t0 = time.time()
        if len(anomaly_events) > 0:
            dom_freq_map, ncc_map, _is_ref_map, dom_freqs4_map = precompute_all_metrics_gpu(
                anomaly_events, windows, windows_ncc, time_arrays,
                n_freq_bins=c.n_freq_bins, freq_chunk=c.freq_chunk,
                ncc_max_lag=c.ncc_max_lag,
            )
        print(f"⏱️  Pré-calcul métriques : {time.time()-t0:.2f}s")

        print("ℹ️  Export plots désactivé.")

        # ── Clustering par types + visualisations enrichies ─────────────
        signal_profile = None
        if len(anomaly_events) > 0:
            t0 = time.time()
            print("\n--- Clustering inter-événements (super-types) ---")
            cluster_labels, _ncc_matrix = cluster_events_by_type(
                anomaly_events, windows_ncc,
                gpu_batch_size=c.gpu_batch_size, ncc_max_lag=c.ncc_max_lag,
                ncc_type_threshold=c.ncc_type_threshold,
            )
            n_types = max(cluster_labels, default=-1) + 1
            print(f"  -> {len(anomaly_events)} événements → {n_types} types "
                  f"(seuil NCC_TYPE={c.ncc_type_threshold})")

            signal_profile = build_signal_profile(
                anomaly_events, windows, time_arrays,
                dom_freq_map, ncc_map, cluster_labels,
                dom_freqs4_map=dom_freqs4_map,
                signal_id=signal_id
            )

            print("--- Association aux types de la base globale (sans filtre passe-bas) ---")
            assign_global_types(
                signal_profile, c.type_db_dir,
                ncc_threshold=c.global_type_ncc_threshold
            )

            print("--- Génération des visualisations enrichies ---")
            ref_plot_paths = plot_reference_windows(
                signal_profile, time_arrays, output_dir, num_workers=c.num_workers
            )

            export_enriched_html(signal_profile, output_dir, ref_plot_paths=ref_plot_paths)
            save_type_windows(signal_profile, output_dir)
            serialize_signal_profile(signal_profile, output_dir)
            print(f"⏱️  Analyse enrichie : {time.time()-t0:.2f}s")

            print("--- Génération du rapport détaillé par type ---")
            type_summaries = plot_all_type_summaries(
                signal_profile, output_dir, bin_width_min=c.type_hist_bin_min
            )
            export_type_summary_html(signal_profile, type_summaries, output_dir)

        print("\n" + "=" * 70)
        print(f"✅ PIPELINE TERMINÉ EN {time.time()-t_start_total:.2f}s")
        print("=" * 70 + "\n")

        return {"output_dir": output_dir, "signal_profile": signal_profile}
