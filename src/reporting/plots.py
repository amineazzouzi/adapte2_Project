"""Génération des plots : fenêtres de référence par événement
(oscillo_analysis.py) et Gantt/comparaisons inter-signaux
(oscillo_correlation.py)."""

import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.signal_processing.frequency import time_seconds_from_axis
from src.core.paths import sig_key_from_signal_id
from src.reporting.palette import TYPE_COLORS, shared_pair_color


def _plot_ref_window_worker(args):
    """Worker : génère et sauvegarde le plot de référence d'un événement."""
    ev_id, cid, signal, time_ax, dom_freq, refs_dir, output_dir = args

    t_rel  = time_seconds_from_axis(time_ax)
    ts_str = pd.to_datetime(time_ax[0]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_rel, signal, color='firebrick', alpha=0.85, linewidth=0.8)
    ax.set_title(
        f"Événement {ev_id:02d} — Type {cid} — Fenêtre de référence\n"
        f"{ts_str}  |  Fréq. dominante : {dom_freq:.1f} Hz"
    )
    ax.set_xlabel("Temps relatif (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()

    fname = f"event_{ev_id:02d}_type{cid}_ref.png"
    fpath = os.path.join(refs_dir, fname)
    fig.savefig(fpath, dpi=100)
    plt.close(fig)
    return ev_id, os.path.relpath(fpath, output_dir)


def plot_reference_windows(signal_profile, time_arrays, output_dir, num_workers=4):
    """
    Génère un plot du signal de référence pour chaque événement.
    Parallélisé via ProcessPoolExecutor (un plot par event).
    Retourne {event_id: rel_path_from_output_dir}.
    """
    events = signal_profile['events']
    if not events:
        return {}

    refs_dir = os.path.join(output_dir, "refs")
    os.makedirs(refs_dir, exist_ok=True)

    tasks = [
        (ev['event_id'], ev['cluster_id'], ev['ref_window'],
         time_arrays[ev['ref_win_idx']], ev['ref_dom_freq'],
         refs_dir, output_dir)
        for ev in events
    ]

    ref_paths = {}
    n_workers = min(num_workers, len(tasks))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for ev_id, rel_path in executor.map(_plot_ref_window_worker, tasks):
            ref_paths[ev_id] = rel_path

    print(f"  -> {len(ref_paths)} plots de référence générés dans 'refs/'")
    return ref_paths


def plot_temporal_overlap(profiles, output_dir, bin_width_min=1):
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

    unique_keys = list(dict.fromkeys(sig_key_from_signal_id(p['signal_id']) for p in profiles))
    cmap        = matplotlib.colormaps['tab10'].resampled(max(len(unique_keys), 1))
    key_to_color = {k: cmap(i) for i, k in enumerate(unique_keys)}
    seen_keys    = set()

    fig, ax = plt.subplots(figsize=(18, 5))

    for profile in profiles:
        all_ts = pd.DatetimeIndex(
            [t for ev in profile['events'] for t in ev['window_timestamps']]
        )
        if len(all_ts) == 0:
            continue
        counts = all_ts.floor(freq).value_counts().sort_index()
        counts = counts.reindex(bins[:-1], fill_value=0)
        bin_centers = counts.index + pd.Timedelta(minutes=bin_width_min / 2)

        key   = sig_key_from_signal_id(profile['signal_id'])
        color = key_to_color[key]
        # Une seule entrée de légende par signal (même couleur pour tous les jours)
        label = key if key not in seen_keys else '_nolegend_'
        seen_keys.add(key)

        ax.fill_between(
            mdates.date2num(bin_centers), counts.values,
            alpha=0.4, color=color, label=label
        )
        ax.plot(mdates.date2num(bin_centers), counts.values,
                color=color, linewidth=1.2)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_xlabel("Date / Heure")
    ax.set_ylabel("Nombre d'anomalies")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"Chevauchement temporel des anomalies — {len(profiles)} profil(s)")
    ax.legend(loc='upper right')
    plt.tight_layout()

    path = os.path.join(output_dir, "temporal_overlap.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Chevauchement temporel : {path}")
    return path


def _fig_pixel_map(fig, ax, path, dpi):
    """
    Calcule le facteur de conversion (coords data -> pixels de l'image PNG
    finale) pour un fig sauvegardé avec bbox_inches='tight'. Ce dernier
    recadre le PNG, donc les pixels finaux ne correspondent pas directement
    à ax.transData ; on corrige via le rapport entre la taille réelle du PNG
    et la tight bbox calculée côté matplotlib (robuste aux arrondis internes).
    Retourne une fonction data_to_px(x_data, y_data) -> (px_x, px_y_top_left).
    """
    from PIL import Image

    fig.canvas.draw()
    pad_inches = 0.1
    tbbox = fig.get_tightbbox(fig.canvas.get_renderer())
    x0_in, y0_in = tbbox.x0 - pad_inches, tbbox.y0 - pad_inches
    width_in  = (tbbox.x1 + pad_inches) - x0_in
    height_in = (tbbox.y1 + pad_inches) - y0_in

    with Image.open(path) as im:
        real_w, real_h = im.size

    scale_x = real_w / (width_in * dpi)
    scale_y = real_h / (height_in * dpi)

    def data_to_px(x_data, y_data):
        disp_x, disp_y = ax.transData.transform((x_data, y_data))
        px = (disp_x - x0_in * dpi) * scale_x
        py_from_bottom = (disp_y - y0_in * dpi) * scale_y
        py_top = real_h - py_from_bottom
        return px, py_top

    return data_to_px


def plot_gantt_timeline(profiles, profiles_with_dirs, output_dir, color_map=None,
                        shared_types=None):
    """
    Diagramme de Gantt : une ligne par signal (boitier/voie),
    barres colorées par type d'anomalie (cluster_id).

    Si 'shared_types' est fourni, une image map HTML est générée : cliquer
    sur une barre dont le type a une correspondance dans un autre signal
    (section 3) saute directement à sa carte de comparaison. Les barres sans
    correspondance partagée ne sont pas cliquables.
    Retourne (chemin_png, html_image_map_ou_None).
    """
    if not profiles:
        return None, None

    # Première paire (signal, type) partagée trouvée pour chaque type -> lien
    pair_to_shared_idx = {}
    for i, st in enumerate(shared_types or []):
        for pkey in [(st['sig_a'], st['cid_a']), (st['sig_b'], st['cid_b'])]:
            pair_to_shared_idx.setdefault(pkey, i)

    # Regrouper les événements par signal (boitier/voie, toutes dates confondues)
    from collections import OrderedDict
    rows: OrderedDict = OrderedDict()
    for profile, _ in profiles_with_dirs:
        key = sig_key_from_signal_id(profile['signal_id'])
        if key not in rows:
            rows[key] = []
        rows[key].extend(profile['events'])

    # Plage temporelle globale
    all_ts = []
    for p in profiles:
        if p['t_start']:
            all_ts.append(p['t_start'])
        if p['t_end']:
            all_ts.append(p['t_end'])
    if not all_ts:
        return None, None

    t_min   = min(all_ts)
    t_max   = max(all_ts)
    total_s = max((t_max - t_min).total_seconds(), 1.0)
    n_rows  = len(rows)

    fig_h = max(2.5, n_rows * 0.9 + 1.8)
    dpi   = 120
    fig, ax = plt.subplots(figsize=(18, fig_h))
    fig.set_dpi(dpi)

    row_labels   = list(rows.keys())
    legend_pairs = {}   # (sig_key, cid) → color, dans l'ordre d'apparition
    clickable_bars = []  # [(x0_data, x1_data, y0_data, y1_data, href), ...]

    for ri, key in enumerate(row_labels):
        for ev in rows[key]:
            cid   = ev['cluster_id']
            pair  = (key, cid)
            color = (color_map.get(pair) if color_map else None) \
                    or TYPE_COLORS[cid % len(TYPE_COLORS)]
            t_ev  = ev['ref_timestamp']
            dur_s = max(ev.get('duration_s', 0), total_s * 0.003, 1.0)
            t_end = t_ev + pd.Timedelta(seconds=dur_s)
            x0    = mdates.date2num(t_ev)
            x1    = mdates.date2num(t_end)

            ax.barh(
                ri,
                x1 - x0,
                left=x0,
                height=0.6,
                color=color,
                alpha=0.90,
            )
            if pair not in legend_pairs:
                legend_pairs[pair] = color

            shared_idx = pair_to_shared_idx.get(pair)
            if shared_idx is not None:
                clickable_bars.append((x0, x1, ri - 0.3, ri + 0.3, shared_idx))

    # Axes
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [k.replace('/', ' / ') for k in row_labels], fontsize=10
    )
    ax.invert_yaxis()
    ax.xaxis_date()
    fmt = '%H:%M' if total_s < 86400 else '%d/%m %H:%M'
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_xlim(mdates.date2num(t_min), mdates.date2num(t_max))
    ax.set_xlabel("durée d'analyse", fontsize=11)
    n_types = len({cid for _, cid in legend_pairs})
    ax.set_title(
        f"Timeline des anomalies — {n_rows} signal(s)  |  {n_types} type(s)",
        fontsize=13,
    )
    ax.grid(axis='x', linestyle='--', alpha=0.35)

    # Légende unifiée : une entrée par (signal, type)
    legend_handles = [
        mpatches.Patch(color=col,
                       label=f"{sig.split('/')[-1]} — T{cid}")
        for (sig, cid), col in sorted(legend_pairs.items())
    ]
    ax.legend(handles=legend_handles, title='Signal / Type', loc='upper right',
              bbox_to_anchor=(1.18, 1), framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, "gantt_timeline.png")
    fig.savefig(path, dpi=dpi, bbox_inches='tight')

    map_html = None
    if clickable_bars:
        data_to_px = _fig_pixel_map(fig, ax, path, dpi)
        areas = []
        for x0, x1, y0, y1, shared_idx in clickable_bars:
            px0, py0 = data_to_px(x0, y0)
            px1, py1 = data_to_px(x1, y1)
            left, right = sorted((px0, px1))
            top, bottom = sorted((py0, py1))
            areas.append(
                f"<area shape='rect' coords='{left:.0f},{top:.0f},{right:.0f},{bottom:.0f}' "
                f"href='#pair-{shared_idx}' "
                f"title='Voir la comparaison — Paire {shared_idx + 1}'/>"
            )
        map_html = "<map name='gantt-map'>" + "".join(areas) + "</map>"

    plt.close(fig)
    print(f"  -> Timeline Gantt : {path}"
          f" ({len(clickable_bars)} barre(s) cliquable(s))")
    return path, map_html


def plot_shared_type_comparisons(shared_types, output_dir):
    """
    Pour chaque paire de types partagés, génère un graphique côte-à-côte
    des fenêtres de référence (signal A à gauche, signal B à droite).
    Sauvegarde dans output_dir/shared_types/shared_pair_NN.png.
    Retourne une liste de chemins RELATIFS à output_dir (ou None si données manquantes).
    """
    shared_dir = os.path.join(output_dir, "shared_types")
    os.makedirs(shared_dir, exist_ok=True)

    paths = []
    for i, st in enumerate(shared_types):
        wa = st.get('window_a')
        wb = st.get('window_b')

        if wa is None and wb is None:
            paths.append(None)
            continue

        color_a = color_b = shared_pair_color(i)

        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        fig.suptitle(
            f"Paire {i + 1} — NCC = {st['ncc']:.3f}",
            fontsize=13, fontweight='bold'
        )

        if wa is not None:
            axes[0].plot(wa, color=color_a, linewidth=0.8)
        axes[0].set_title(
            f"{st['sig_a']} — Type {st['cid_a']}\n"
            f"Fréq. moy. {st['freq_a']:.1f} Hz  |  {st['count_a']} événement(s)"
        )
        axes[0].set_xlabel("Échantillons")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)

        if wb is not None:
            axes[1].plot(wb, color=color_b, linewidth=0.8)
        axes[1].set_title(
            f"{st['sig_b']} — Type {st['cid_b']}\n"
            f"Fréq. moy. {st['freq_b']:.1f} Hz  |  {st['count_b']} événement(s)"
        )
        axes[1].set_xlabel("Échantillons")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"shared_pair_{i:02d}.png"
        fig.savefig(os.path.join(shared_dir, fname), dpi=120, bbox_inches='tight')
        plt.close(fig)
        paths.append(os.path.join("shared_types", fname))

    n_ok = sum(1 for p in paths if p)
    print(f"  -> {n_ok} graphique(s) comparatif(s) générés dans 'shared_types/'")
    return paths
