"""
Écriture/lecture du signal_profile — le schéma partagé entre
oscillo_analysis.py (écrit signal_profile.json + refs/type{cid}_window.npy)
et oscillo_correlation.py (lit les deux pour comparer les signaux entre eux).
"""

import json
import os

import numpy as np
import pandas as pd


def save_type_windows(signal_profile, output_dir):
    """
    Sauvegarde la fenêtre de référence du premier événement de chaque type
    sous refs/type{cid}_window.npy — utilisé par oscillo_correlation.py
    pour le calcul de similarité inter-signaux.
    """
    refs_dir = os.path.join(output_dir, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    seen_cids = set()
    for ev in signal_profile['events']:
        cid = ev['cluster_id']
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        arr = np.asarray(ev['ref_window'], dtype=np.float32)
        np.save(os.path.join(refs_dir, f"type{cid}_window.npy"), arr)
    print(f"  -> {len(seen_cids)} fenêtre(s) de type sauvegardée(s) dans 'refs/'")


def serialize_signal_profile(signal_profile, output_dir):
    """Sérialise le profil (sans les arrays numpy) en JSON pour oscillo_correlation.py."""
    def ts(t):
        return pd.Timestamp(t).isoformat() if t is not None else None

    data = {
        'signal_id':     signal_profile['signal_id'],
        'total_windows': signal_profile['total_windows'],
        'n_clusters':    signal_profile['n_clusters'],
        't_start':       ts(signal_profile['t_start']),
        't_end':         ts(signal_profile['t_end']),
        'events': [
            {
                'event_id':          ev['event_id'],
                'cluster_id':        ev['cluster_id'],
                'type_label':        ev.get('type_label', f"Type {ev['cluster_id']}"),
                'ref_timestamp':     ts(ev['ref_timestamp']),
                'ref_dom_freq':      ev['ref_dom_freq'],
                'ref_dom_freqs4':    ev.get('ref_dom_freqs4', [ev['ref_dom_freq'], 0.0, 0.0, 0.0]),
                'window_count':      ev['window_count'],
                'duration_s':        ev['duration_s'],
                'window_timestamps': [ts(t) for t in ev['window_timestamps']],
                'dom_freqs':         ev['dom_freqs'],
                'ncc_vs_ref':        ev['ncc_vs_ref'],
            }
            for ev in signal_profile['events']
        ],
    }

    path = os.path.join(output_dir, "signal_profile.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"--- Profil JSON sérialisé : '{path}' ---")
    return path


def load_signal_profile(profile_dir):
    """Charge un signal_profile.json (écrit par serialize_signal_profile) et
    reconvertit les timestamps en pd.Timestamp. Utilisé par
    oscillo_correlation.py."""
    json_path = os.path.join(profile_dir, "signal_profile.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Profil introuvable : {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    for ev in raw['events']:
        ev['ref_timestamp']     = pd.Timestamp(ev['ref_timestamp'])
        ev['window_timestamps'] = [pd.Timestamp(t) for t in ev['window_timestamps']]

    raw['t_start'] = pd.Timestamp(raw['t_start']) if raw['t_start'] else None
    raw['t_end']   = pd.Timestamp(raw['t_end'])   if raw['t_end']   else None

    # Fallback : dériver depuis les événements si les champs sont absents du JSON
    if raw['t_start'] is None and raw['events']:
        raw['t_start'] = min(ev['ref_timestamp'] for ev in raw['events'])
    if raw['t_end'] is None and raw['events']:
        raw['t_end'] = max(
            ev['window_timestamps'][-1]
            for ev in raw['events'] if ev['window_timestamps']
        )

    return raw
