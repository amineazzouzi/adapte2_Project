"""Rapports HTML — rapport enrichi par signal (oscillo_analysis.py).
Le rapport de corrélation (oscillo_correlation.py) est ajouté ici en Phase 3."""

import os
from collections import defaultdict

import numpy as np


def export_enriched_html(signal_profile, base_output_dir, ref_plot_paths=None):
    """
    Rapport HTML : sections 5 (détail événements) et 6 (fenêtres de référence).
    """
    events    = signal_profile['events']
    by_cluster = defaultdict(list)
    for ev in events:
        by_cluster[ev['cluster_id']].append(ev)

    html = [
        "<html><head><meta charset='utf-8'>",
        "<title>Rapport Enrichi — Analyse des Anomalies</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#f5f6fa}",
        "h1,h2,h3{color:#2c3e50}",
        ".card{background:#fff;margin-bottom:24px;padding:20px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.12)}",
        ".badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#3498db;color:#fff;margin:2px}",
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px;margin-top:8px}",
        "table{border-collapse:collapse;width:100%}",
        "th,td{border:1px solid #ddd;padding:8px 12px;text-align:left;font-size:13px}",
        "th{background:#2c3e50;color:#fff}",
        "tr:nth-child(even){background:#f9f9f9}",
        "</style></head><body>",
        f"<h1>Rapport d'Analyse Enrichi — {signal_profile['signal_id']}</h1>",
    ]

    # 5. Tableau détail événements
    html += ["<div class='card'>", "<h2>5. Détail des événements</h2>",
             "<table>",
             "<tr><th>#</th><th>Type</th><th>Début</th><th>Fenêtres</th>"
             "<th>Durée (s)</th><th>Fréq. moy. (Hz)</th><th>NCC moy. vs réf.</th></tr>"]
    for ev in events:
        mf  = np.mean(ev['dom_freqs']) if ev['dom_freqs'] else 0.0
        mn  = np.mean(ev['ncc_vs_ref']) if ev['ncc_vs_ref'] else 1.0
        ts  = ev['ref_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        html.append(
            f"<tr><td>{ev['event_id']}</td>"
            f"<td><span class='badge'>Type {ev['cluster_id']}</span></td>"
            f"<td>{ts}</td><td>{ev['window_count']}</td>"
            f"<td>{ev['duration_s']:.1f}</td><td>{mf:.1f}</td><td>{mn:.4f}</td></tr>"
        )
    html += ["</table>", "</div>"]

    # 6. Fenêtres de référence par événement
    if ref_plot_paths:
        html += ["<div class='card'>",
                 "<h2>6. Fenêtres de référence par événement</h2>"]
        for cid in sorted(by_cluster):
            c_events = sorted(by_cluster[cid], key=lambda e: e['event_id'])
            html.append(
                f"<h3>Type {cid} &mdash; {len(c_events)} événement(s)</h3>"
            )
            html.append(
                "<div style='display:grid;grid-template-columns:"
                "repeat(auto-fit,minmax(480px,1fr));gap:16px'>"
            )
            for ev in c_events:
                ev_id = ev['event_id']
                if ev_id not in ref_plot_paths:
                    continue
                ts = ev['ref_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                freqs4 = ev.get('ref_dom_freqs4', [ev['ref_dom_freq'], 0.0, 0.0, 0.0])
                freqs4_str = ', '.join(f"{f:.1f}" for f in freqs4 if f > 0) or f"{ev['ref_dom_freq']:.1f}"
                html.append(
                    f"<div style='border:1px solid #ddd;border-radius:6px;"
                    f"padding:12px;background:#fafafa'>"
                    f"<p style='margin:0 0 6px'>"
                    f"<strong>Événement {ev_id}</strong> &bull; {ts} &bull; "
                    f"{ev['window_count']} fen&ecirc;tres &bull; "
                    f"Dur&eacute;e&nbsp;: {ev['duration_s']:.1f}&nbsp;s &bull; "
                    f"Fr&eacute;q.&nbsp;dominantes&nbsp;: {freqs4_str}&nbsp;Hz</p>"
                    f"<img src='{ref_plot_paths[ev_id]}' "
                    f"alt='R&eacute;f&eacute;rence &eacute;v&eacute;nement {ev_id}'/>"
                    f"</div>"
                )
            html.append("</div>")  # close grid
        html.append("</div>")  # close card

    html += ["</body></html>"]

    path = os.path.join(base_output_dir, "rapport_enrichi.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"--- Rapport enrichi : '{path}' ---")
    return path


def export_type_summary_html(signal_profile, type_summaries, output_dir):
    """
    Rapport HTML dédié — une ligne par type détecté :
    fenêtre de référence | nombre de fenêtres | histogramme d'apparition
    (bins de 10 min) sur toute la durée d'analyse.
    Fichier séparé de rapport_enrichi.html, ne le remplace pas.
    """
    html = [
        "<html><head><meta charset='utf-8'>",
        "<title>Rapport par Type — Analyse des Anomalies</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#f5f6fa}",
        "h1{color:#2c3e50}",
        ".card{background:#fff;margin-bottom:24px;padding:20px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.12)}",
        "table{border-collapse:collapse;width:100%}",
        "th,td{border:1px solid #ddd;padding:10px;text-align:center;vertical-align:middle;font-size:13px}",
        "th{background:#2c3e50;color:#fff}",
        "tr:nth-child(even){background:#f9f9f9}",
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px}",
        ".count{font-size:28px;font-weight:bold;color:#2c3e50}",
        "</style></head><body>",
        f"<h1>Rapport par Type — {signal_profile['signal_id']}</h1>",
        "<div class='card'>",
        "<table>",
        "<tr><th>Fenêtre de référence</th><th>Nombre de fenêtres</th>"
        "<th>Apparition dans le temps (bins 10 min)</th></tr>",
    ]

    for cid in sorted(type_summaries):
        ref_rel, hist_rel, count = type_summaries[cid]
        html.append(
            f"<tr>"
            f"<td><img src='{ref_rel}' alt='Type {cid} — référence'/></td>"
            f"<td class='count'>{count}</td>"
            f"<td><img src='{hist_rel}' alt='Type {cid} — histogramme'/></td>"
            f"</tr>"
        )

    html += ["</table>", "</div>", "</body></html>"]

    path = os.path.join(output_dir, "rapport_types_detaille.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"--- Rapport par type : '{path}' ---")
    return path


def export_correlation_html(profiles, overlap_path, gantt_path, output_dir,
                            ncc_type_threshold, shared_types=None, shared_plot_paths=None,
                            gantt_map_html=None):
    """Rapport HTML : section 1 (chevauchement), 2 (Gantt), 3 (types communs cliquables)."""
    from src.reporting.palette import shared_pair_color

    def rel(p):
        if not p:
            return None
        try:
            return os.path.relpath(p, output_dir)
        except ValueError:
            return p

    css = """
html{scroll-behavior:smooth}
body{font-family:Arial,sans-serif;margin:20px;background:#f5f6fa}
h1,h2,h3{color:#2c3e50}
.card{background:#fff;margin-bottom:24px;padding:20px;border-radius:8px;
      box-shadow:0 2px 6px rgba(0,0,0,.12)}
img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px;margin-top:6px}
.badge-grid{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 24px}
.pair-badge{
  display:flex;align-items:center;gap:7px;
  background:#ecf0f1;border-radius:20px;padding:8px 14px;
  text-decoration:none;color:#2c3e50;font-size:13px;font-weight:600;
  box-shadow:0 1px 3px rgba(0,0,0,.15);transition:transform .15s,box-shadow .15s
}
.pair-badge:hover{transform:translateY(-2px);box-shadow:0 4px 10px rgba(0,0,0,.22)}
.cdot{width:16px;height:16px;border-radius:50%;flex-shrink:0;display:inline-block;
      border:1px solid rgba(0,0,0,.12)}
.ncc-pill{background:#2c3e50;color:#fff;border-radius:10px;
          padding:2px 8px;font-size:11px;margin:0 2px}
.pair-card{scroll-margin-top:60px}
.pair-images{display:flex;gap:20px;margin-top:14px}
.pair-images .side{flex:1;text-align:center}
.side-label{font-weight:bold;font-size:14px;margin-bottom:6px;
            display:flex;align-items:center;gap:6px;justify-content:center}
.side-meta{font-size:12px;color:#555;margin-top:6px}
.back-link{display:inline-block;margin-top:14px;font-size:12px;
           color:#3498db;text-decoration:none}
.back-link:hover{text-decoration:underline}
.no-shared{color:#888;font-style:italic}
"""

    html = [
        "<html><head><meta charset='utf-8'>",
        "<title>Rapport de Corrélation Inter-Signaux</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Rapport de Corrélation Inter-Signaux</h1>",
    ]

    # ── Section 1 : chevauchement temporel ───────────────────────────
    if overlap_path:
        html += [
            "<div class='card'>",
            "<h2>1 — Chevauchement temporel des anomalies</h2>",
            f"<img src='{rel(overlap_path)}' alt='Chevauchement temporel'/>",
            "</div>",
        ]

    # ── Section 2 : timeline Gantt par type ──────────────────────────
    if gantt_path:
        usemap = " usemap='#gantt-map'" if gantt_map_html else ""
        html += [
            "<div class='card'>",
            "<h2>2 — Timeline des anomalies par type</h2>",
        ]
        if gantt_map_html:
            html.append(
                "<p style='font-size:12px;color:#888'>"
                "Les barres avec une correspondance dans un autre signal sont "
                "cliquables et renvoient vers leur comparaison (section 3).</p>"
            )
        html.append(f"<img src='{rel(gantt_path)}' alt='Timeline Gantt'{usemap}/>")
        if gantt_map_html:
            html.append(gantt_map_html)
        html.append("</div>")

    # ── Section 3 : types communs ─────────────────────────────────────
    html += [
        "<div class='card' id='section-types-communs'>",
        "<h2>3 — Types communs entre signaux</h2>",
    ]

    if not shared_types:
        html.append(
            f"<p class='no-shared'>Aucun type partagé détecté "
            f"(seuil NCC &ge; {ncc_type_threshold:.2f}).</p>"
        )
    else:
        # ── Grille de badges cliquables ──────────────────────────────
        html.append(
            "<p>Cliquez sur une paire pour accéder directement "
            "au graphique comparatif&nbsp;:</p>"
        )
        html.append("<div class='badge-grid'>")
        for i, st in enumerate(shared_types):
            color_a = color_b = shared_pair_color(i)
            html.append(
                f"<a class='pair-badge' href='#pair-{i}'>"
                f"<span class='cdot' style='background:{color_a}'></span>"
                f"{st['sig_a']}&nbsp;T{st['cid_a']}"
                f"<span class='ncc-pill'>NCC&nbsp;{st['ncc']:.2f}</span>"
                f"{st['sig_b']}&nbsp;T{st['cid_b']}"
                f"<span class='cdot' style='background:{color_b}'></span>"
                f"</a>"
            )
        html.append("</div>")

        # ── Cartes détaillées ────────────────────────────────────────
        for i, st in enumerate(shared_types):
            color_a = color_b = shared_pair_color(i)
            plot_rel = (shared_plot_paths[i]
                        if shared_plot_paths and i < len(shared_plot_paths)
                        else None)
            img_a_rel = (rel(st['img_a'])
                         if st.get('img_a') and os.path.exists(st['img_a']) else None)
            img_b_rel = (rel(st['img_b'])
                         if st.get('img_b') and os.path.exists(st['img_b']) else None)
            sig_a = st['sig_a']
            sig_b = st['sig_b']
            cid_a = st['cid_a']
            cid_b = st['cid_b']

            html.append(f"<div class='card pair-card' id='pair-{i}'>")
            html.append(
                f"<h3>"
                f"Paire {i + 1}&nbsp;&nbsp;"
                f"<span style='color:{color_a}'>&#9679;</span>&nbsp;"
                f"{sig_a} Type {cid_a}"
                f"&nbsp;&nbsp;&#8596;&nbsp;&nbsp;"
                f"<span style='color:{color_b}'>&#9679;</span>&nbsp;"
                f"{sig_b} Type {cid_b}"
                f"&nbsp;&nbsp;&mdash;&nbsp;&nbsp;NCC = {st['ncc']:.3f}"
                f"</h3>"
            )

            # Graphique matplotlib côte-à-côte
            if plot_rel:
                html.append(
                    f"<img src='{plot_rel}' "
                    f"alt='Comparaison types partagés paire {i + 1}'/>"
                )

            # Images de référence originales (si disponibles)
            if img_a_rel or img_b_rel:
                html.append("<div class='pair-images'>")

                html.append("<div class='side'>")
                html.append(
                    f"<div class='side-label'>"
                    f"<span class='cdot' style='background:{color_a}'></span>"
                    f"{sig_a} — Type {cid_a}</div>"
                )
                if img_a_rel:
                    html.append(
                        f"<img src='{img_a_rel}' "
                        f"alt='Référence {sig_a} Type {cid_a}'/>"
                    )
                html.append(
                    f"<div class='side-meta'>"
                    f"Fréq. moy. : <strong>{st['freq_a']:.1f} Hz</strong>"
                    f" &nbsp;|&nbsp; {st['count_a']} événement(s)</div>"
                )
                html.append("</div>")

                html.append("<div class='side'>")
                html.append(
                    f"<div class='side-label'>"
                    f"<span class='cdot' style='background:{color_b}'></span>"
                    f"{sig_b} — Type {cid_b}</div>"
                )
                if img_b_rel:
                    html.append(
                        f"<img src='{img_b_rel}' "
                        f"alt='Référence {sig_b} Type {cid_b}'/>"
                    )
                html.append(
                    f"<div class='side-meta'>"
                    f"Fréq. moy. : <strong>{st['freq_b']:.1f} Hz</strong>"
                    f" &nbsp;|&nbsp; {st['count_b']} événement(s)</div>"
                )
                html.append("</div>")

                html.append("</div>")  # end .pair-images

            html.append(
                "<a class='back-link' href='#section-types-communs'>"
                "&#8679; Retour au sommaire des types communs</a>"
            )
            html.append("</div>")  # end .pair-card

    html.append("</div>")  # end section card
    html.append("</body></html>")

    path = os.path.join(output_dir, "rapport_correlation.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"--- Rapport corrélation : '{path}' ---")
    return path
