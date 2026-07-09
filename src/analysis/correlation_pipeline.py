"""Détection de types d'anomalies partagés entre signaux + orchestrateur
de la corrélation inter-signaux (voir oscillo_correlation.py)."""

import os
import time
from collections import defaultdict

import numpy as np

from src.core.paths import sig_key_from_signal_id, output_dir_for
from src.io.profile_io import load_signal_profile
from src.signal_processing.ncc_cpu import max_ncc_full_range
from src.reporting.palette import build_color_map
from src.reporting.plots import (
    plot_temporal_overlap, plot_gantt_timeline, plot_shared_type_comparisons,
)
from src.reporting.html_report import export_correlation_html


def _cluster_summary_for_key(profiles_with_dirs, target_key):
    """
    Agrège les clusters de tous les jours d'un même signal (boitier/voie).
    Retourne {cluster_id: {'mean_freq', 'count', 'ref_img', 'window'}}.
    'window' est la fenêtre numpy chargée depuis refs/type{cid}_window.npy
    (None si le fichier n'existe pas).
    """
    cluster_freqs = defaultdict(list)
    cluster_imgs  = {}
    cluster_wins  = {}

    for profile, pdir in profiles_with_dirs:
        if sig_key_from_signal_id(profile['signal_id']) != target_key:
            continue
        refs_dir = os.path.join(pdir, 'refs')
        for ev in profile['events']:
            cid = ev['cluster_id']
            cluster_freqs[cid].append(ev['ref_dom_freq'])
            if cid not in cluster_imgs and os.path.isdir(refs_dir):
                for fname in sorted(os.listdir(refs_dir)):
                    if f'_type{cid}_ref.png' in fname:
                        cluster_imgs[cid] = os.path.join(refs_dir, fname)
                        break
            if cid not in cluster_wins:
                npy = os.path.join(refs_dir, f'type{cid}_window.npy')
                if os.path.exists(npy):
                    cluster_wins[cid] = np.load(npy).astype(np.float64)

    return {
        cid: {
            'mean_freq': sum(freqs) / len(freqs),
            'count':     len(freqs),
            'ref_img':   cluster_imgs.get(cid),
            'window':    cluster_wins.get(cid),
        }
        for cid, freqs in cluster_freqs.items()
    }


def find_shared_types(profiles_with_dirs, ncc_threshold):
    """
    Compare les fenêtres de référence de chaque type entre signaux distincts
    via NCC (Normalized Cross-Correlation). Deux types sont 'partagés' si
    leur NCC maximale dépasse ncc_threshold.
    Retourne une liste triée par NCC décroissante.
    """
    unique_keys = list(dict.fromkeys(
        sig_key_from_signal_id(p['signal_id']) for p, _ in profiles_with_dirs
    ))
    if len(unique_keys) < 2:
        return []

    summaries = {
        key: _cluster_summary_for_key(profiles_with_dirs, key)
        for key in unique_keys
    }

    shared = []
    seen   = set()
    for i in range(len(unique_keys)):
        for j in range(i + 1, len(unique_keys)):
            key_a, key_b = unique_keys[i], unique_keys[j]
            for cid_a, info_a in summaries[key_a].items():
                for cid_b, info_b in summaries[key_b].items():
                    wa = info_a.get('window')
                    wb = info_b.get('window')
                    if wa is None or wb is None:
                        continue
                    # Aligner les longueurs (tronquer au minimum)
                    n = min(len(wa), len(wb))
                    ncc_val = max_ncc_full_range(wa[:n], wb[:n])
                    if ncc_val < ncc_threshold:
                        continue
                    pair_key = (key_a, cid_a, key_b, cid_b)
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    shared.append({
                        'ncc':      ncc_val,
                        'sig_a':    key_a,   'cid_a':   cid_a,
                        'freq_a':   info_a['mean_freq'],
                        'count_a':  info_a['count'],
                        'img_a':    info_a['ref_img'],
                        'window_a': wa,
                        'sig_b':    key_b,   'cid_b':   cid_b,
                        'freq_b':   info_b['mean_freq'],
                        'count_b':  info_b['count'],
                        'img_b':    info_b['ref_img'],
                        'window_b': wb,
                    })

    shared.sort(key=lambda x: x['ncc'], reverse=True)
    print(f"  -> {len(shared)} type(s) partagé(s) "
          f"(NCC ≥ {ncc_threshold:.2f} sur {sum(len(s) for s in summaries.values())} types)")
    return shared


class CorrelationPipeline:
    """
    Charge les signal_profile.json de plusieurs signaux, détecte les types
    d'anomalies partagés entre eux (NCC entre fenêtres de référence), et
    exporte les visualisations (chevauchement temporel, Gantt, comparaisons)
    + le rapport HTML de corrélation.
    """

    def __init__(self, config):
        self.config = config

    def run(self):
        c = self.config
        t_global = time.time()

        print("=" * 70)
        print("PIPELINE CORRÉLATION INTER-SIGNAUX")
        print(f"  Signaux sélectionnés : {len(c.signals)}")
        for s in c.signals:
            print(f"    - {s['boitier']} / voie_{s['voie']}")
        print(f"  Fenêtre co-occurrence : ±{c.corr_window_s}s")
        print("=" * 70)

        output_dir = os.path.join(c.project_dir, c.output_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        # Chargement des profils
        print(f"\nChargement de {len(c.signals)} profil(s)...")
        profiles           = []
        profiles_with_dirs = []   # [(profile, profile_dir), ...]
        for sig in c.signals:
            d = os.path.join(c.project_dir, output_dir_for(sig['boitier'], sig['voie']))
            try:
                p = load_signal_profile(d)
                profiles.append(p)
                profiles_with_dirs.append((p, d))
                print(f"  OK : {p['signal_id']}  ({len(p['events'])} événements, "
                      f"{p['n_clusters']} types, {p['total_windows']} fenêtres)")
            except Exception as e:
                print(f"  ERREUR {d} : {e}")

        if len(profiles) < 2:
            print("\nAu moins 2 profils valides requis. Arrêt.")
            return None

        # Types partagés — calculés en premier pour pouvoir unifier les couleurs
        print("\nRecherche des types partagés entre signaux...")
        shared_types = find_shared_types(profiles_with_dirs, c.ncc_type_threshold)

        # Mapping de couleurs — une couleur par type, utilisé pour la timeline Gantt
        print("\nConstruction du mapping de couleurs...")
        pairs = []
        seen  = set()
        for profile, _ in profiles_with_dirs:
            key = sig_key_from_signal_id(profile['signal_id'])
            for ev in profile['events']:
                pair = (key, ev['cluster_id'])
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        pairs.sort()
        color_map = build_color_map(pairs)
        print(f"  -> {len(color_map)} paire(s) (signal, type) → couleur")

        # Visualisations
        print("\nGénération des visualisations...")
        overlap_path = plot_temporal_overlap(profiles, output_dir, bin_width_min=c.hist_bin_min)

        # Timeline Gantt — une couleur par type, barres cliquables vers leur
        # comparaison partagée (section 3) le cas échéant
        print("\nGénération de la timeline Gantt...")
        gantt_path, gantt_map_html = plot_gantt_timeline(
            profiles, profiles_with_dirs, output_dir,
            color_map=color_map, shared_types=shared_types
        )

        # Graphiques comparatifs des types partagés — couleur commune par paire
        shared_plot_paths = []
        if shared_types:
            print("\nGénération des graphiques comparatifs...")
            shared_plot_paths = plot_shared_type_comparisons(shared_types, output_dir)

        # Rapport HTML
        print("\nExport du rapport HTML...")
        report_path = export_correlation_html(
            profiles, overlap_path, gantt_path, output_dir,
            c.ncc_type_threshold,
            shared_types=shared_types,
            shared_plot_paths=shared_plot_paths,
            gantt_map_html=gantt_map_html,
        )

        print(f"\n{'='*70}")
        print(f"TERMINÉ EN {time.time()-t_global:.2f}s")
        print(f"Résultats : {os.path.abspath(output_dir)}/rapport_correlation.html")
        print(f"{'='*70}\n")

        return {"output_dir": output_dir, "report_path": report_path}
