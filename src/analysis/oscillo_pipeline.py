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
from src.signal_processing.windowing import (
    filter_anomaly_windows, filter_by_peak_threshold, classify_windows_by_peak_kmeans,
    compute_window_energy,
)
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

        # ── Classification par nombre de pics isolés (parmi les anomalies) ──
        # Dépend du filtrage de l'étape précédente : seules les fenêtres
        # retenues comme anomalies (is_anomaly) passent la segmentation +
        # KMeans sur max(signal**4) par segment qui classe pic_1 / pic_2 /
        # pic_3 / ... — voir classify_windows_by_peak_kmeans. Les fenêtres
        # non-anomalies gardent peak_label="" sans passer par le KMeans.
        peak_counts = np.zeros(len(windows), dtype=int)
        peak_labels = np.full(len(windows), "", dtype=object)
        anomaly_idx = np.flatnonzero(is_anomaly) if len(is_anomaly) > 0 else np.array([], dtype=int)
        n_classified = 0
        if len(anomaly_idx) > 0:
            t0 = time.time()
            peak_counts[anomaly_idx], peak_labels[anomaly_idx] = classify_windows_by_peak_kmeans(
                windows[anomaly_idx],
                search_width=c.peak_kmeans_search_width,
                min_cluster_separation=c.peak_kmeans_min_cluster_separation,
            )
            n_classified = int(np.sum(peak_labels != ""))
            print(f"Fenêtres classées par nombre de pics : {n_classified} / {len(anomaly_idx)} anomalies "
                  f"(segments={c.peak_kmeans_search_width} éch., "
                  f"séparation min={c.peak_kmeans_min_cluster_separation}x)")
            print(f"⏱️  Classification pics : {time.time()-t0:.2f}s")

        # ── Énergie des fenêtres "pic" (windowing.py::compute_window_energy) ──
        # Parmi les fenêtres classées "pic" ci-dessus, celles dont l'énergie
        # (signal brut, filtre médian avant somme) dépasse peak_energy_threshold
        # comptent EN PLUS comme "pic + forme" — flag additionnel, ne remplace
        # pas la classification pic_N. Seuil validé dans pic_kmeans_explorer.ipynb.
        if len(windows) > 0:
            peak_energies = compute_window_energy(
                windows, median_filter_size=c.peak_energy_median_filter_size
            )
            is_pic_forme = (peak_labels != "") & (peak_energies > c.peak_energy_threshold)
            n_pic_forme = int(np.sum(is_pic_forme))
            print(f"Fenêtres 'pic + forme' (énergie > {c.peak_energy_threshold}) : "
                  f"{n_pic_forme} / {n_classified}")
        else:
            peak_energies = np.array([], dtype=float)
            is_pic_forme = np.array([], dtype=bool)

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

        # ── Isolement des fenêtres "pic" (avant tout calcul de similarité) ──
        # Une fenêtre classée pic_N (voir classify_windows_by_peak_kmeans) est
        # exclue du groupement temporel NCC et deviendra directement un
        # événement à elle seule plus bas : on connaît déjà son type par le
        # comptage de pics, donc aucune comparaison de ressemblance (NCC)
        # n'est nécessaire pour elle.
        if len(is_anomaly) > 0:
            is_pic_window = peak_labels != ""
            regular_is_anomaly = is_anomaly.copy()
            regular_is_anomaly[is_pic_window] = 0
        else:
            is_pic_window = np.array([], dtype=bool)
            regular_is_anomaly = is_anomaly

        # ── Groupement temporel par NCC (GPU) — fenêtres non-pic uniquement ──
        t0 = time.time()
        if len(regular_is_anomaly) > 0:
            regular_events = track_events_temporal_gpu(
                windows_ncc, regular_is_anomaly, timestamps,
                threshold=c.ncc_threshold, max_lag=c.ncc_max_lag
            )
            # Fenêtre de référence de chaque événement = celle qui a le plus
            # grand max (signal brut) parmi toutes les fenêtres de l'événement
            # — pas forcément la première chronologiquement.
            for event in regular_events:
                event["ref_idx"] = max(event["indices"], key=lambda i: windows[i].max())
            print(f"\nTotal groupes (événements) NCC : {len(regular_events)}")
        else:
            regular_events = []
            print("Aucune fenêtre à traiter.")
        print(f"⏱️  Groupement NCC : {time.time()-t0:.2f}s")

        # ── Événements "pic" : une fenêtre isolée = un événement, sans NCC ──
        pic_events = []
        if len(is_pic_window) > 0:
            for i in np.flatnonzero(is_pic_window):
                pic_events.append({
                    "debut": timestamps[i], "fin": timestamps[i],
                    "indices": [int(i)], "sims": [], "ref_idx": int(i),
                    "peak_label": str(peak_labels[i]),
                    "peak_energy": float(peak_energies[i]),
                    "peak_forme": bool(is_pic_forme[i]),
                })
            if pic_events:
                print(f"Total événements pic (isolés, sans NCC) : {len(pic_events)}")

        # ── Pré-calcul GPU des métriques (fréquence dominante + NCC vs réf.) ──
        # Événements réguliers UNIQUEMENT : les événements "pic" n'ont besoin
        # ni de fréquence dominante (un pic isolé n'a pas de fréquence
        # significative à extraire) ni de NCC vs référence (événement à une
        # seule fenêtre, qui EST sa propre référence) — les deux calculs sont
        # sautés pour eux. build_signal_profile retombe sur 0.0 par défaut
        # pour ces champs (dom_freq_map.get(idx, 0.0)).
        all_events = regular_events + pic_events
        all_events.sort(key=lambda ev: ev["debut"])

        dom_freq_map, ncc_map, dom_freqs4_map = {}, {}, {}
        t0 = time.time()
        if len(regular_events) > 0:
            dom_freq_map, ncc_map, _is_ref_map, dom_freqs4_map = precompute_all_metrics_gpu(
                regular_events, windows, windows_ncc, time_arrays,
                ncc_max_lag=c.ncc_max_lag,
            )
        print(f"⏱️  Pré-calcul métriques : {time.time()-t0:.2f}s")

        print("ℹ️  Export plots désactivé.")

        # ── Clustering par types + visualisations enrichies ─────────────
        signal_profile = None
        if len(all_events) > 0:
            t0 = time.time()
            print("\n--- Clustering inter-événements (super-types), fenêtres régulières uniquement ---")
            regular_cluster_labels, _ncc_matrix = cluster_events_by_type(
                regular_events, windows_ncc,
                gpu_batch_size=c.gpu_batch_size, ncc_max_lag=c.ncc_max_lag,
                ncc_type_threshold=c.ncc_type_threshold,
            )
            n_regular_types = max(regular_cluster_labels, default=-1) + 1
            print(f"  -> {len(regular_events)} événement(s) régulier(s) → {n_regular_types} type(s) "
                  f"(seuil NCC_TYPE={c.ncc_type_threshold})")

            # Tous les événements "pic" (quel que soit leur nombre de pics
            # isolés, pic_1 / pic_2 / pic_3 / …) sont regroupés dans un seul
            # et même type "type : pic", numéroté à la suite des types NCC
            # réguliers — pas de calcul NCC inter-événements pour ce type,
            # déjà connu par construction.
            PIC_TYPE_LABEL = "type : pic"
            pic_cid = n_regular_types
            pic_label_to_cid = {
                lbl: pic_cid for lbl in {ev["peak_label"] for ev in pic_events}
            }
            if pic_label_to_cid:
                print(f"  -> {len(pic_events)} événement(s) pic → 1 type "
                      f"({PIC_TYPE_LABEL}, regroupant {', '.join(sorted(pic_label_to_cid))})")

            # cluster_labels / type_labels doivent suivre le même ordre que
            # all_events (trié par date de début, cf. plus haut) — on les
            # reconstruit événement par événement plutôt que de concaténer
            # les deux listes régulier/pic dans leur ordre d'origine.
            regular_cid_by_id = dict(zip((id(ev) for ev in regular_events), regular_cluster_labels))
            cluster_labels, type_labels = [], []
            for ev in all_events:
                if "peak_label" in ev:
                    cluster_labels.append(pic_cid)
                    type_labels.append(PIC_TYPE_LABEL)
                else:
                    cid = regular_cid_by_id[id(ev)]
                    cluster_labels.append(cid)
                    type_labels.append(f"Type {cid}")

            signal_profile = build_signal_profile(
                all_events, windows, time_arrays,
                dom_freq_map, ncc_map, cluster_labels,
                dom_freqs4_map=dom_freqs4_map,
                signal_id=signal_id,
                peak_counts=peak_counts, peak_labels=peak_labels,
                peak_energies=peak_energies, peak_forme=is_pic_forme,
                type_labels=type_labels,
            )

            print("--- Association aux types de la base globale (sans filtre passe-bas) ---")
            assign_global_types(
                signal_profile, c.type_db_dir,
                ncc_threshold=c.global_type_ncc_threshold,
                skip_cluster_ids=set(pic_label_to_cid.values()),
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
