#!/usr/bin/env python
# coding: utf-8
"""
Pipeline de corrélation inter-signaux.

Lit les signal_profile.json produits par oscillo_analysis.py et détecte
les co-occurrences d'anomalies entre les signaux sélectionnés.

Usage :
  1. Lancer oscillo_analysis.py pour chaque signal avec un OUTPUT_DIR distinct.
  2. Renseigner PROFILE_DIRS ci-dessous avec ces dossiers.
  3. python oscillo_correlation.py
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION — À ADAPTER PAR L'UTILISATEUR
# ══════════════════════════════════════════════════════════════════════

# Répertoire racine du projet (là où oscillo_analysis.py génère ses outputs)
PROJECT_DIR = "/home/aazzouzi/Projet IA/Adapte2/adapte2_project"

# Signaux à corréler : choisir les boîtiers et voies manuellement.
# Chaque entrée correspond à un run d'oscillo_analysis.py déjà effectué.
# Le dossier de sortie est auto-calculé comme :
#   {PROJECT_DIR}/outputs_{boitier}_v{voie}_{date}
SIGNALS_TO_CORRELATE = [
    {"boitier": "boitier_1", "voie": 1, "date": "2026-02-17"},
    {"boitier": "boitier_2", "voie": 1, "date": "2026-02-17"},
    # Ajouter d'autres voies ici, ex :
    # {"boitier": "boitier_1", "voie": 2, "date": "2026-02-17"},
]

OUTPUT_DIR    = os.path.join(PROJECT_DIR, "outputs_correlation")
CORR_WINDOW_S = 30.0   # Fenêtre de co-occurrence (secondes)
HIST_BIN_MIN  = 30     # Largeur des bins pour les histogrammes synchronisés (minutes)


def _profile_dir(sig):
    """Retrouve le dossier de sortie d'oscillo_analysis.py pour un signal donné."""
    return os.path.join(PROJECT_DIR, f"outputs_{sig['boitier']}_v{sig['voie']}_{sig['date']}")


# ══════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════

def load_signal_profile(profile_dir):
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


# ══════════════════════════════════════════════════════════════════════
# CORRÉLATION INTER-SIGNAUX
# ══════════════════════════════════════════════════════════════════════

def find_cooccurrences(profiles, corr_window_s=CORR_WINDOW_S):
    """
    Détecte les co-occurrences entre événements de signaux différents.
    Deux événements co-occurrent si |t_ref_A - t_ref_B| <= corr_window_s.
    Seules les fenêtres de référence sont comparées (une par événement).
    """
    cooccurrences = []
    n = len(profiles)

    for i in range(n):
        for j in range(i + 1, n):
            pa, pb = profiles[i], profiles[j]
            for ev_a in pa['events']:
                t_a = ev_a['ref_timestamp']
                for ev_b in pb['events']:
                    delta = (ev_b['ref_timestamp'] - t_a).total_seconds()
                    if abs(delta) <= corr_window_s:
                        cooccurrences.append({
                            'signal_a':   pa['signal_id'],
                            'signal_b':   pb['signal_id'],
                            'event_a_id': ev_a['event_id'],
                            'event_b_id': ev_b['event_id'],
                            'cluster_a':  ev_a['cluster_id'],
                            'cluster_b':  ev_b['cluster_id'],
                            'ref_ts_a':   t_a,
                            'ref_ts_b':   ev_b['ref_timestamp'],
                            'delta_s':    delta,
                        })

    return cooccurrences


def build_cooccurrence_matrices(profiles, cooccurrences):
    """
    Matrice [type_A × type_B] = nombre de co-occurrences pour chaque paire de signaux.
    """
    matrices = {}
    pairs = defaultdict(list)
    for co in cooccurrences:
        pairs[(co['signal_a'], co['signal_b'])].append(co)

    for (sig_a, sig_b), cos in pairs.items():
        pa     = next(p for p in profiles if p['signal_id'] == sig_a)
        pb     = next(p for p in profiles if p['signal_id'] == sig_b)
        n_a    = pa['n_clusters']
        n_b    = pb['n_clusters']
        mat    = np.zeros((n_a, n_b), dtype=int)
        for co in cos:
            mat[co['cluster_a'], co['cluster_b']] += 1
        matrices[(sig_a, sig_b)] = mat

    return matrices


# ══════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════

def plot_sync_timeline(profiles, cooccurrences, output_dir):
    """
    Un axe par signal (vue Gantt empilée), avec des lignes verticales rouges
    marquant les co-occurrences détectées. Permet de voir la synchronicité.
    """
    n = len(profiles)
    t_min = min(p['t_start'] for p in profiles if p['t_start'])
    t_max = max(p['t_end']   for p in profiles if p['t_end'])

    fig, axes = plt.subplots(n, 1, figsize=(18, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    sig_ax = {p['signal_id']: axes[i] for i, p in enumerate(profiles)}

    for i, profile in enumerate(profiles):
        ax     = axes[i]
        n_clust = profile['n_clusters']
        cmap   = plt.cm.get_cmap('Set2', max(n_clust, 1))

        for ev in profile['events']:
            t_left   = mdates.date2num(ev['ref_timestamp'])
            dur_days = max(ev['duration_s'] / 86400.0, 1.0 / 86400.0)
            ax.barh(0, dur_days, left=t_left, height=0.6,
                    color=cmap(ev['cluster_id']), alpha=0.85)

        ax.set_ylabel(profile['signal_id'], fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(mdates.date2num(t_min), mdates.date2num(t_max))
        ax.grid(True, axis='x', alpha=0.3)

        seen, handles = set(), []
        for ev in profile['events']:
            cid = ev['cluster_id']
            if cid not in seen:
                seen.add(cid)
                handles.append(mpatches.Patch(color=cmap(cid), label=f"Type {cid}"))
        ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=min(len(handles), 6))

    # Marquer les co-occurrences sur les deux axes concernés
    for co in cooccurrences:
        ax_a = sig_ax.get(co['signal_a'])
        ax_b = sig_ax.get(co['signal_b'])
        if ax_a:
            ax_a.axvline(mdates.date2num(co['ref_ts_a']),
                         color='#e74c3c', linewidth=0.7, alpha=0.5)
        if ax_b:
            ax_b.axvline(mdates.date2num(co['ref_ts_b']),
                         color='#e74c3c', linewidth=0.7, alpha=0.5)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    axes[-1].set_xlabel("Heure")

    fig.suptitle(
        f"Timeline synchronisée — {n} signaux\n"
        f"{len(cooccurrences)} co-occurrences détectées (fenêtre ±{CORR_WINDOW_S}s)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    path = os.path.join(output_dir, "timeline_sync.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Timeline synchronisée : {path}")
    return path


def plot_cooccurrence_matrices(matrices, output_dir):
    """Heatmap de la matrice de co-occurrences pour chaque paire de signaux."""
    paths = {}

    for (sig_a, sig_b), mat in matrices.items():
        if mat.sum() == 0:
            continue

        rows, cols = mat.shape
        fig, ax = plt.subplots(figsize=(max(5, cols * 1.2 + 2), max(4, rows * 1.2 + 1.5)))

        im = ax.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0)
        plt.colorbar(im, ax=ax, label='Nb co-occurrences', fraction=0.046, pad=0.04)

        ax.set_xticks(range(cols))
        ax.set_xticklabels([f"Type {j}" for j in range(cols)])
        ax.set_yticks(range(rows))
        ax.set_yticklabels([f"Type {i}" for i in range(rows)])

        for i in range(rows):
            for j in range(cols):
                if mat[i, j] > 0:
                    ax.text(j, i, str(mat[i, j]), ha='center', va='center',
                            fontweight='bold', fontsize=11,
                            color='white' if mat[i, j] > mat.max() * 0.6 else 'black')

        short_a = sig_a.split('/')[-1] if '/' in sig_a else sig_a
        short_b = sig_b.split('/')[-1] if '/' in sig_b else sig_b
        ax.set_xlabel(f"Types — {short_b}")
        ax.set_ylabel(f"Types — {short_a}")
        ax.set_title(f"Co-occurrences (±{CORR_WINDOW_S}s)\n{short_a}  ×  {short_b}")

        plt.tight_layout()
        fname = (f"cooccurrence_"
                 f"{sig_a.replace('/', '_').replace(' ', '_')}_"
                 f"{sig_b.replace('/', '_').replace(' ', '_')}.png")
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        paths[(sig_a, sig_b)] = path
        print(f"  -> Matrice co-occurrence : {path}")

    return paths


def plot_lag_distribution(cooccurrences, output_dir):
    """Distribution des délais δt entre co-occurrences."""
    if not cooccurrences:
        return None

    deltas = [co['delta_s'] for co in cooccurrences]
    median_d = np.median(deltas)
    mean_d   = np.mean(deltas)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(deltas, bins=min(30, len(deltas)),
            color='#3498db', edgecolor='white', alpha=0.85)
    ax.axvline(0,        color='#e74c3c',  linestyle='--', linewidth=1.5, label='Simultané (δt=0)')
    ax.axvline(median_d, color='#f39c12',  linestyle=':',  linewidth=1.5, label=f'Médiane ({median_d:+.1f}s)')
    ax.axvline(mean_d,   color='#2ecc71',  linestyle=':',  linewidth=1.5, label=f'Moyenne ({mean_d:+.1f}s)')
    ax.set_xlabel("Délai δt (s)   [négatif = signal B précède signal A]")
    ax.set_ylabel("Nb de co-occurrences")
    ax.set_title(f"Distribution des délais de co-occurrence  (n={len(deltas)})")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "lag_distribution.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Distribution des délais : {path}")
    return path


def plot_temporal_overlap(profiles, output_dir, bin_width_min=HIST_BIN_MIN):
    """
    Histogrammes temporels superposés de tous les signaux sur la même grille.
    Révèle les périodes où plusieurs signaux présentent des anomalies simultanément.
    """
    if not profiles:
        return None

    t_min = min(p['t_start'] for p in profiles if p['t_start'])
    t_max = max(p['t_end']   for p in profiles if p['t_end'])
    freq  = f'{bin_width_min}min'
    bins  = pd.date_range(t_min.floor(freq), t_max.ceil(freq), freq=freq)

    colors = plt.cm.get_cmap('tab10', len(profiles))
    fig, ax = plt.subplots(figsize=(18, 5))

    for idx, profile in enumerate(profiles):
        all_ts = pd.DatetimeIndex(
            [t for ev in profile['events'] for t in ev['window_timestamps']]
        )
        if len(all_ts) == 0:
            continue
        counts = all_ts.floor(freq).value_counts().sort_index()
        counts = counts.reindex(bins[:-1], fill_value=0)
        bin_centers = counts.index + pd.Timedelta(minutes=bin_width_min / 2)

        # Normaliser par le max pour pouvoir comparer des signaux d'échelles différentes
        max_c = counts.max()
        norm  = counts / max_c if max_c > 0 else counts

        ax.fill_between(
            mdates.date2num(bin_centers), norm.values,
            alpha=0.4, color=colors(idx), label=profile['signal_id']
        )
        ax.plot(mdates.date2num(bin_centers), norm.values,
                color=colors(idx), linewidth=1.2)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_xlabel("Heure")
    ax.set_ylabel("Densité d'anomalies (normalisée)")
    ax.set_title(f"Chevauchement temporel des anomalies — {len(profiles)} signaux")
    ax.legend(loc='upper right')
    plt.tight_layout()

    path = os.path.join(output_dir, "temporal_overlap.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Chevauchement temporel : {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# RAPPORT HTML
# ══════════════════════════════════════════════════════════════════════

def export_correlation_html(profiles, cooccurrences, matrices,
                             timeline_path, matrix_paths, lag_path,
                             overlap_path, output_dir):
    pair_cos = defaultdict(list)
    for co in cooccurrences:
        pair_cos[(co['signal_a'], co['signal_b'])].append(co)

    def rel(p):
        return os.path.relpath(p, output_dir) if p else None

    n_events_total = sum(len(p['events']) for p in profiles)

    html = [
        "<html><head><meta charset='utf-8'>",
        "<title>Rapport de Corrélation Inter-Signaux</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#f5f6fa}",
        "h1,h2,h3{color:#2c3e50}",
        ".card{background:#fff;margin-bottom:24px;padding:20px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.12)}",
        ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:20px}",
        ".kpi{background:#2c3e50;color:#fff;padding:16px;border-radius:8px;text-align:center}",
        ".kpi .val{font-size:28px;font-weight:bold;color:#e74c3c}",
        ".kpi .lbl{font-size:12px;margin-top:4px}",
        ".badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#3498db;color:#fff;margin:2px}",
        ".badge-b{background:#e67e22}",
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px;margin-top:8px}",
        "table{border-collapse:collapse;width:100%}",
        "th,td{border:1px solid #ddd;padding:8px 12px;text-align:left;font-size:13px}",
        "th{background:#2c3e50;color:#fff}",
        "tr:nth-child(even){background:#f9f9f9}",
        "</style></head><body>",
        "<h1>Rapport de Corrélation Inter-Signaux</h1>",
    ]

    # KPIs
    html += [
        "<div class='card'><div class='kpi-grid'>",
        f"<div class='kpi'><div class='val'>{len(profiles)}</div><div class='lbl'>Signaux analysés</div></div>",
        f"<div class='kpi'><div class='val'>{n_events_total}</div><div class='lbl'>Événements totaux</div></div>",
        f"<div class='kpi'><div class='val'>{len(cooccurrences)}</div><div class='lbl'>Co-occurrences (±{CORR_WINDOW_S}s)</div></div>",
        "</div>",
        "<table><tr><th>Signal</th><th>Événements</th><th>Types</th><th>Fenêtres</th></tr>",
    ]
    for p in profiles:
        html.append(f"<tr><td>{p['signal_id']}</td><td>{len(p['events'])}</td>"
                    f"<td>{p['n_clusters']}</td><td>{p['total_windows']}</td></tr>")
    html += ["</table>", "</div>"]

    # 1. Timeline synchronisée
    html += ["<div class='card'>", "<h2>1. Timeline synchronisée</h2>"]
    if timeline_path:
        html.append(f"<img src='{rel(timeline_path)}' alt='Timeline synchronisée'/>")
    html.append("</div>")

    # 2. Chevauchement temporel
    if overlap_path:
        html += ["<div class='card'>", "<h2>2. Chevauchement temporel des anomalies</h2>",
                 f"<img src='{rel(overlap_path)}' alt='Chevauchement temporel'/>",
                 "</div>"]

    # 3. Distribution des délais
    if lag_path:
        deltas = [co['delta_s'] for co in cooccurrences]
        html += ["<div class='card'>", "<h2>3. Distribution des délais de co-occurrence</h2>",
                 f"<img src='{rel(lag_path)}' alt='Distribution délais'/>"]
        if deltas:
            html.append(
                f"<p>Médiane : <strong>{np.median(deltas):+.1f}s</strong> | "
                f"Moyenne : <strong>{np.mean(deltas):+.1f}s</strong> | "
                f"Délai max : <strong>{max(abs(d) for d in deltas):.1f}s</strong></p>"
            )
        html.append("</div>")

    # 4. Matrices de co-occurrence
    html += ["<div class='card'>", "<h2>4. Matrices de co-occurrence par paire</h2>"]
    for key, mat_path in matrix_paths.items():
        sig_a, sig_b = key
        n_co_pair = len(pair_cos.get(key, []))
        html.append(f"<h3>{sig_a} × {sig_b}  —  {n_co_pair} co-occurrences</h3>")
        html.append(f"<img src='{rel(mat_path)}' alt='Matrice co-occurrence'/>")
    html.append("</div>")

    # 5. Tableau détaillé (limité à 300 lignes)
    if cooccurrences:
        html += ["<div class='card'>", "<h2>5. Détail des co-occurrences</h2>",
                 "<table>",
                 "<tr><th>Signal A</th><th>Type A</th><th>Timestamp A</th>"
                 "<th>Signal B</th><th>Type B</th><th>Timestamp B</th><th>δt (s)</th></tr>"]
        shown = sorted(cooccurrences, key=lambda x: x['ref_ts_a'])[:300]
        for co in shown:
            html.append(
                f"<tr>"
                f"<td>{co['signal_a']}</td>"
                f"<td><span class='badge'>Type {co['cluster_a']}</span></td>"
                f"<td>{co['ref_ts_a'].strftime('%H:%M:%S')}</td>"
                f"<td>{co['signal_b']}</td>"
                f"<td><span class='badge badge-b'>Type {co['cluster_b']}</span></td>"
                f"<td>{co['ref_ts_b'].strftime('%H:%M:%S')}</td>"
                f"<td>{co['delta_s']:+.1f}</td>"
                f"</tr>"
            )
        if len(cooccurrences) > 300:
            html.append(f"<tr><td colspan='7'><em>... et {len(cooccurrences)-300} autres</em></td></tr>")
        html += ["</table>", "</div>"]

    html.append("</body></html>")

    path = os.path.join(output_dir, "rapport_correlation.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"--- Rapport corrélation : '{path}' ---")
    return path


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time as _time
    t_global = _time.time()

    print("=" * 70)
    print("PIPELINE CORRÉLATION INTER-SIGNAUX")
    print(f"  Signaux sélectionnés : {len(SIGNALS_TO_CORRELATE)}")
    for s in SIGNALS_TO_CORRELATE:
        print(f"    - {s['boitier']} / voie_{s['voie']} / {s['date']}")
    print(f"  Fenêtre co-occurrence : ±{CORR_WINDOW_S}s")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Chargement des profils
    print(f"\nChargement de {len(SIGNALS_TO_CORRELATE)} profil(s)...")
    profiles = []
    for sig in SIGNALS_TO_CORRELATE:
        d = _profile_dir(sig)
        try:
            p = load_signal_profile(d)
            profiles.append(p)
            print(f"  OK : {p['signal_id']}  ({len(p['events'])} événements, "
                  f"{p['n_clusters']} types, {p['total_windows']} fenêtres)")
        except Exception as e:
            print(f"  ERREUR {d} : {e}")

    if len(profiles) < 2:
        print("\nAu moins 2 profils valides requis. Arrêt.")
        exit(1)

    # Corrélation
    print(f"\nDétection des co-occurrences (±{CORR_WINDOW_S}s)...")
    cooccurrences = find_cooccurrences(profiles, corr_window_s=CORR_WINDOW_S)
    print(f"  -> {len(cooccurrences)} co-occurrences détectées")

    matrices = build_cooccurrence_matrices(profiles, cooccurrences)

    # Visualisations
    print("\nGénération des visualisations...")
    timeline_path = plot_sync_timeline(profiles, cooccurrences, OUTPUT_DIR)
    matrix_paths  = plot_cooccurrence_matrices(matrices, OUTPUT_DIR)
    lag_path      = plot_lag_distribution(cooccurrences, OUTPUT_DIR)
    overlap_path  = plot_temporal_overlap(profiles, OUTPUT_DIR)

    # Rapport HTML
    print("\nExport du rapport HTML...")
    export_correlation_html(
        profiles, cooccurrences, matrices,
        timeline_path, matrix_paths, lag_path, overlap_path,
        OUTPUT_DIR
    )

    print(f"\n{'='*70}")
    print(f"TERMINÉ EN {_time.time()-t_global:.2f}s")
    print(f"Résultats : {os.path.abspath(OUTPUT_DIR)}/rapport_correlation.html")
    print(f"{'='*70}\n")
