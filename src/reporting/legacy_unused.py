"""
Code hérité, NON APPELÉ par le pipeline actuel (oscillo_analysis.py:
`__main__` fait explicitement l'impasse dessus — "Export plots désactivé").
Conservé tel quel (déplacé, pas réécrit) pour ne pas perdre de fonctionnalité
qui pourrait être réactivée plus tard, mais ne fait partie d'aucun chemin
d'exécution vérifié : ne pas modifier sans re-brancher et re-tester.
"""

import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

from src.core.config import SignalConfig
from src.signal_processing.frequency import time_seconds_from_axis
from src.analysis.event_tracking import precompute_all_metrics_gpu
from src.reporting.palette import TYPE_COLORS


def _plot_one_window(args):
    """Worker exécuté dans un process séparé : trace + sauvegarde une fenêtre."""
    (win_idx, num_ev, i_in_ev, signal, time_axis,
     dom_freq, ncc, is_ref, event_dir, folder_name) = args

    freq_str = f"Freq Dom: {dom_freq:.1f}Hz"

    if is_ref:
        score_str = f"REFERENCE_{dom_freq:.1f}Hz"
        score_title = f"(référence | {freq_str})"
    else:
        score_str = f"ncc_{ncc:.4f}_{dom_freq:.1f}Hz"
        score_title = f"NCC = {ncc:.4f} | {freq_str}"

    t_rel = time_seconds_from_axis(time_axis)
    ts_str = pd.to_datetime(time_axis[0]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_rel, signal, color='firebrick', alpha=0.85,
            linewidth=0.8, label=f'Signal ({freq_str})')

    ax.set_title(
        f"Événement {num_ev:02d} | Fenêtre #{win_idx} | "
        f"{ts_str} | {score_title}"
    )
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    plt.xticks(rotation=15)
    ax.legend(loc="upper right")
    plt.tight_layout()

    filename = f"window_{win_idx:04d}_{score_str}.png"
    full_img_path = os.path.join(event_dir, filename)
    plt.savefig(full_img_path, dpi=100)
    plt.close(fig)

    ref_image_path = f"{folder_name}/{filename}" if is_ref else None

    return {
        "win_idx": win_idx,
        "num_ev": num_ev,
        "dom_freq": dom_freq,
        "filename": filename,
        "ref_image_path": ref_image_path,
    }


def save_anomaly_event_plots_gpu(anomaly_events, windows, time_arrays,
                                  base_output_dir="outputs",
                                  num_workers=4,
                                  dom_freq_map=None, ncc_map=None, is_ref_map=None):
    """
    Version GPU-accélérée : les métriques (NCC, freq dominante) sont
    pré-calculées en BATCH sur GPU avant la boucle, et l'export des
    images est parallélisé sur plusieurs processus CPU.
    Accepte des maps pré-calculées pour éviter de les recalculer.
    """
    import shutil

    if not anomaly_events:
        print("Aucun événement à exporter.")
        return dom_freq_map or {}, ncc_map or {}, is_ref_map or {}

    if os.path.exists(base_output_dir):
        shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"--- Sauvegarde dans : '{os.path.abspath(base_output_dir)}' ---")

    # 1️⃣ Pré-calcul GPU (ou réutilisation des maps fournies)
    if dom_freq_map is None or ncc_map is None or is_ref_map is None:
        _cfg = SignalConfig()
        dom_freq_map, ncc_map, is_ref_map = precompute_all_metrics_gpu(
            anomaly_events, windows, time_arrays,
            n_freq_bins=_cfg.n_freq_bins, freq_chunk=_cfg.freq_chunk,
            ncc_max_lag=_cfg.ncc_max_lag,
        )

    # 2️⃣ Préparer les dossiers + arguments des workers
    tasks = []
    event_meta = []

    for num_ev, event in enumerate(anomaly_events):
        indices = event["indices"]

        t_debut_val = time_arrays[indices[0]][0]
        t_fin_val = time_arrays[indices[-1]][-1]

        t_debut_str_clean = pd.to_datetime(t_debut_val).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        t_fin_str_clean = pd.to_datetime(t_fin_val).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        folder_nom_debut = t_debut_str_clean.replace(":", "-").replace(".", "_").replace(" ", "_")
        folder_name = f"evenement_{num_ev:02d}_debut_{folder_nom_debut}"
        event_dir = os.path.join(base_output_dir, folder_name)
        os.makedirs(event_dir, exist_ok=True)

        event_meta.append({
            "num_ev": num_ev,
            "folder_name": folder_name,
            "t_debut_str_clean": t_debut_str_clean,
            "t_fin_str_clean": t_fin_str_clean,
            "indices": indices,
        })

        for i_in_ev, win_idx in enumerate(indices):
            if not is_ref_map[win_idx]:
                continue
            tasks.append((
                win_idx, num_ev, i_in_ev,
                windows[win_idx], time_arrays[win_idx],
                dom_freq_map[win_idx], ncc_map[win_idx], is_ref_map[win_idx],
                event_dir, folder_name
            ))

    print(f"🚀 Export de {len(tasks)} plots (fenêtres ref uniquement, {num_workers} workers)...")
    t0 = time.time()

    results_by_event = {num_ev: {"dom_freqs": [], "ref_image_path": None}
                         for num_ev in range(len(anomaly_events))}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_plot_one_window, task) for task in tasks]
        for future in as_completed(futures):
            res = future.result()
            results_by_event[res["num_ev"]]["dom_freqs"].append(res["dom_freq"])
            if res["ref_image_path"]:
                results_by_event[res["num_ev"]]["ref_image_path"] = res["ref_image_path"]

    elapsed = time.time() - t0
    print(f"✅ Export terminé en {elapsed:.2f}s ({len(tasks)/elapsed:.1f} plots/s)")

    # 3️⃣ Construction du récapitulatif CSV + HTML (identique à l'original)
    recap_data = []
    html_content = [
        "<html><head><meta charset='utf-8'>",
        "<title>Rapport des Anomalies</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f6fa; }",
        "h1 { color: #333; }",
        ".anomaly-card { background: #fff; margin-bottom: 20px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".header-info { display: flex; flex-direction: column; gap: 5px; margin-bottom: 15px; }",
        "img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }",
        ".stat { font-weight: bold; color: #e74c3c; }",
        "</style>",
        "</head><body>",
        f"<h1>Rapport d'Analyse : {len(anomaly_events)} événements détectés</h1>"
    ]

    for meta in event_meta:
        num_ev = meta["num_ev"]
        res = results_by_event[num_ev]
        moy_freq = round(np.mean(res["dom_freqs"]), 2) if res["dom_freqs"] else 0.0

        recap_data.append({
            "anomalie_id": f"anomalie {num_ev + 1}",
            "date_de_debut": meta["t_debut_str_clean"],
            "date_de_fin": meta["t_fin_str_clean"],
            "nb_de_fenetres": len(meta["indices"]),
            "moyenne_frequence_dominante_Hz": moy_freq
        })

        html_content.append("<div class='anomaly-card'>")
        html_content.append(f"<h2>Anomalie {num_ev + 1}</h2>")
        html_content.append("<div class='header-info'>")
        html_content.append(f"<span><strong>Date de début :</strong> {meta['t_debut_str_clean']}</span>")
        html_content.append(f"<span><strong>Date de fin :</strong> {meta['t_fin_str_clean']}</span>")
        html_content.append(f"<span><strong>Nombre de fenêtres :</strong> <span class='stat'>{len(meta['indices'])}</span></span>")
        html_content.append(f"<span><strong>Moyenne de fréquence dominante :</strong> <span class='stat'>{moy_freq} Hz</span></span>")
        html_content.append("</div>")
        if res["ref_image_path"]:
            html_content.append("<h3>Plot de la fenêtre de référence :</h3>")
            html_content.append(f"<img src='{res['ref_image_path']}' alt='Reference Window' />")
        html_content.append("</div>")

    html_content.append("</body></html>")

    df_recap = pd.DataFrame(recap_data)
    recap_csv_path = os.path.join(base_output_dir, "recapitulatif_anomalies.csv")
    df_recap.to_csv(recap_csv_path, index=False, sep=";")

    recap_html_path = os.path.join(base_output_dir, "rapport_anomalies.html")
    with open(recap_html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))

    print(f"\n--- Export terminé : {len(anomaly_events)} événements sauvegardés ---")
    print(f"--- Récapitulatif CSV : '{recap_csv_path}' ---")
    print(f"--- Rapport HTML : '{recap_html_path}' ---")

    return dom_freq_map, ncc_map, is_ref_map


def plot_global_timeline(signal_profile, output_dir):
    """
    Vue Gantt : un rectangle par événement, coloré par type.
    Permet de voir d'un coup d'œil la distribution temporelle de chaque type.
    """
    events = signal_profile['events']
    if not events:
        return None

    n_clust = signal_profile['n_clusters']
    palette = TYPE_COLORS
    if n_clust > len(palette):
        import matplotlib.colors as mcolors
        extra_cmap = plt.cm.get_cmap('tab20')
        palette = palette + [mcolors.to_hex(extra_cmap(i)) for i in range(20)]
    color_for = lambda cid: palette[cid % len(palette)]

    fig, ax = plt.subplots(figsize=(18, max(4, len(events) * 0.22 + 2)))

    for ev in events:
        t_left   = mdates.date2num(ev['ref_timestamp'])
        dur_days = max(ev['duration_s'] / 86400.0, 1.0 / 86400.0)
        ax.barh(ev['event_id'], dur_days, left=t_left, height=0.7,
                color=color_for(ev['cluster_id']), alpha=0.85)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_xlabel("Heure")
    ax.set_ylabel("Événement #")
    ax.set_title(
        f"Timeline globale — {signal_profile['signal_id']}\n"
        f"{len(events)} événements  |  {n_clust} types  |  "
        f"{signal_profile['total_windows']} fenêtres totales"
    )

    seen, handles = set(), []
    for ev in events:
        cid = ev['cluster_id']
        if cid not in seen:
            seen.add(cid)
            handles.append(mpatches.Patch(color=color_for(cid), label=f"Type {cid}"))
    ax.legend(handles=sorted(handles, key=lambda h: int(h.get_label().split()[-1])),
              loc='upper right', bbox_to_anchor=(1.12, 1))

    plt.tight_layout()
    path = os.path.join(output_dir, "timeline_globale.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_type_histogram_worker(args):
    """Worker : génère et sauvegarde l'histogramme temporel d'un type d'anomalie."""
    (cid, window_timestamps_flat, dom_freqs_flat,
     t_start, t_end, bin_width_min, n_events, types_dir) = args

    all_ts    = pd.DatetimeIndex(window_timestamps_flat)
    freq      = f'{bin_width_min}min'
    counts    = all_ts.floor(freq).value_counts().sort_index()
    full_bins = pd.date_range(t_start.floor(freq), t_end.ceil(freq), freq=freq)
    counts    = counts.reindex(full_bins, fill_value=0)

    mean_c  = counts.mean()
    std_c   = counts.std()
    burst_t = mean_c + 2.0 * std_c

    bin_centers = counts.index + pd.Timedelta(minutes=bin_width_min / 2)
    bar_colors  = ['#e74c3c' if v > burst_t else '#3498db' for v in counts.values]

    total_w     = int(counts.sum())
    active_bins = int((counts > 0).sum())
    recur_rate  = active_bins / max(len(counts), 1)
    p = counts.values / total_w if total_w > 0 else counts.values
    p = p[p > 0]
    entropy   = (-np.sum(p * np.log2(p)) / np.log2(max(len(counts), 2))
                 if len(p) > 0 else 0.0)
    peak_time = (counts.idxmax() + pd.Timedelta(minutes=bin_width_min / 2)
                 ).strftime('%H:%M') if total_w > 0 else 'N/A'
    med_freq  = float(np.median(dom_freqs_flat)) if dom_freqs_flat else 0.0

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(mdates.date2num(bin_centers), counts.values,
           width=bin_width_min / 1440.0,
           color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.4)
    if burst_t > 0:
        ax.axhline(burst_t, color='#e74c3c', linestyle='--', linewidth=1.2,
                   label=f'Seuil burst (μ+2σ = {burst_t:.1f})')

    ax.set_title(
        f"Type {cid}  |  {n_events} événements  |  {total_w} fenêtres  |  "
        f"Fréq. dominante médiane : {med_freq:.1f} Hz\n"
        f"Récurrence : {recur_rate:.0%}  |  Pic : {peak_time}  |  "
        f"Entropie temporelle : {entropy:.2f}"
    )
    ax.set_xlabel("Heure d'apparition")
    ax.set_ylabel("Nombre de fenêtres")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    if burst_t > 0:
        ax.legend()

    plt.tight_layout()
    path = os.path.join(types_dir, f"type_{cid:02d}_histogram.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return cid, path


def plot_type_histograms(signal_profile, output_dir, bin_width_min=30, num_workers=4):
    """
    Pour chaque type d'anomalie : histogramme temporel (x=heure, y=nb fenêtres).
    Les bins en rouge dépassent le seuil burst (moyenne + 2σ).
    Parallélisé par type via ProcessPoolExecutor.
    """
    events  = signal_profile['events']
    t_start = signal_profile['t_start']
    t_end   = signal_profile['t_end']
    if not events or t_start is None:
        return {}

    types_dir = os.path.join(output_dir, "types")
    os.makedirs(types_dir, exist_ok=True)

    by_cluster = defaultdict(list)
    for ev in events:
        by_cluster[ev['cluster_id']].append(ev)

    tasks = []
    for cid, c_events in sorted(by_cluster.items()):
        tasks.append((
            cid,
            [t for ev in c_events for t in ev['window_timestamps']],
            [f for ev in c_events for f in ev['dom_freqs']],
            t_start, t_end, bin_width_min, len(c_events), types_dir
        ))

    histogram_paths = {}
    n_workers = min(num_workers, len(tasks))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for cid, path in executor.map(_plot_type_histogram_worker, tasks):
            histogram_paths[cid] = path

    return histogram_paths


def plot_ncc_matrix(ncc_matrix, cluster_labels, output_dir, max_events_ncc_matrix=150):
    """
    Heatmap NCC entre toutes les fenêtres de référence, triée par type.
    Skippée si trop d'événements (illisible au-delà de max_events_ncc_matrix).
    """
    N = len(cluster_labels)
    if N <= 1:
        return None
    if N > max_events_ncc_matrix:
        print(f"  Matrice NCC ignorée ({N} > {max_events_ncc_matrix} événements — illisible)")
        return None

    order        = np.argsort(cluster_labels)
    ncc_sorted   = ncc_matrix[np.ix_(order, order)]
    labels_sorted = np.array(cluster_labels)[order]

    cell_size = max(0.35, min(0.6, 8.0 / N))
    fig_size  = max(6, N * cell_size)
    fig, ax   = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(ncc_sorted, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, label='NCC', fraction=0.046, pad=0.04)

    # Frontières entre clusters
    boundaries = [0]
    prev = labels_sorted[0]
    for i, lbl in enumerate(labels_sorted[1:], 1):
        if lbl != prev:
            boundaries.append(i)
            prev = lbl
    boundaries.append(N)
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='black', linewidth=1.5)
        ax.axvline(b - 0.5, color='black', linewidth=1.5)

    ax.set_title(f"Matrice NCC inter-références ({N} événements, triée par type)")
    ax.set_xlabel("Événement #")
    ax.set_ylabel("Événement #")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(output_dir, "ncc_matrix_events.png")
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return path
