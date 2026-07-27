"""
Base de types persistante — GLOBALE, partagée entre tous les boîtiers/voies/
dates, stockée hors de results/outputs_* (jamais effacée par un run,
contrairement aux dossiers de sortie par signal). Grossit au fil des runs :
dès qu'une forme de signal ne correspond à aucun type déjà connu (NCC en
dessous du seuil), elle devient un nouveau type.

Comparaison sur le signal BRUT, SANS le filtre passe-bas (voir
signal_processing.filtering) : contrairement au clustering intra-run (voir
clustering.py), qui compare des fenêtres du même signal/run pour le suivi
d'événements, ici on compare des fenêtres venant potentiellement de
runs/jours/capteurs différents — on ne veut pas perdre de détail qui
distingue deux types réels.

Deux dossiers, mêmes ID en titre de fichier :
  type_db_dir/raw_signals/type_0007.npy
  type_db_dir/plots/type_0007.png
"""

import fcntl
import os
import re
from collections import defaultdict

import numpy as np

from src.signal_processing.ncc_cpu import max_ncc_full_range
from src.reporting.plots import plot_type_database_entry

_ID_RE = re.compile(r"type_(\d+)\.npy$")


def _existing_type_ids(raw_dir):
    if not os.path.isdir(raw_dir):
        return []
    ids = [m.group(1) for f in os.listdir(raw_dir) if (m := _ID_RE.match(f))]
    return sorted(int(i) for i in ids)


def match_or_register_type(ref_window, type_db_dir, ncc_threshold):
    """
    Compare ref_window (signal brut, sans filtre passe-bas) à tous les types
    déjà connus de la base. Si la NCC max >= ncc_threshold, renvoie l'ID du
    type existant le plus proche. Sinon, enregistre un nouveau type (raw
    .npy + plot, même ID dans le nom de fichier) et renvoie son nouvel ID.
    Retourne (type_id: int, is_new: bool).

    Verrouillée (fcntl.flock) sur toute la section lecture-décision-écriture :
    plusieurs runs de scripts/oscillo_analysis.py peuvent tourner en
    parallèle (un par GPU, voir src/analysis/batch_pipeline.py) et
    partagent cette même base — sans verrou, deux process pourraient
    décider indépendamment qu'une forme est nouvelle et s'écraser
    mutuellement le même ID.
    """
    raw_dir   = os.path.join(type_db_dir, "raw_signals")
    plots_dir = os.path.join(type_db_dir, "plots")
    os.makedirs(raw_dir, exist_ok=True)

    ref_window = np.asarray(ref_window, dtype=np.float64)

    lock_path = os.path.join(type_db_dir, ".lock")
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            existing_ids = _existing_type_ids(raw_dir)

            best_id, best_ncc = None, -1.0
            for tid in existing_ids:
                known = np.load(os.path.join(raw_dir, f"type_{tid:04d}.npy"))
                n = min(len(ref_window), len(known))
                ncc_val = max_ncc_full_range(ref_window[:n], known[:n])
                if ncc_val > best_ncc:
                    best_id, best_ncc = tid, ncc_val

            if best_id is not None and best_ncc >= ncc_threshold:
                return best_id, False

            new_id = (max(existing_ids) + 1) if existing_ids else 0
            np.save(os.path.join(raw_dir, f"type_{new_id:04d}.npy"),
                    ref_window.astype(np.float32))
            plot_type_database_entry(new_id, ref_window, plots_dir)
            return new_id, True
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def assign_global_types(signal_profile, type_db_dir, ncc_threshold):
    """
    Pour chaque type LOCAL détecté dans ce run (cluster_id du clustering
    intra-signal, voir clustering.cluster_events_by_type), compare sa
    fenêtre de référence brute à la base de types globale et l'enregistre
    si elle est nouvelle. Retourne {local_cluster_id: global_type_id}.
    """
    events = signal_profile['events']
    if not events:
        return {}

    by_cluster = defaultdict(list)
    for ev in events:
        by_cluster[ev['cluster_id']].append(ev)

    mapping = {}
    n_new = 0
    for cid in sorted(by_cluster):
        ref_event = min(by_cluster[cid], key=lambda e: e['ref_timestamp'])
        global_id, is_new = match_or_register_type(
            ref_event['ref_window'], type_db_dir, ncc_threshold
        )
        mapping[cid] = global_id
        n_new += int(is_new)
        print(f"  Type local {cid} -> type global {global_id:04d} "
              f"({'NOUVEAU' if is_new else 'existant'})")

    print(f"  -> {n_new} nouveau(x) type(s) ajouté(s) à la base "
          f"({len(by_cluster) - n_new} déjà connu(s))")
    return mapping
