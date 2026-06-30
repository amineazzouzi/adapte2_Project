"""Rapports HTML interactifs Plotly pour un signal et pour la corrélation inter-signaux."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from .summary import generate_summary


def _fig_to_html(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def generate_dashboard(
    signal_profile: dict,
    output_dir: Path,
    waveform_images: dict | None = None,
) -> str:
    """Génère un rapport HTML interactif Plotly.

    Sections : résumé texte, Gantt timeline, heatmap horaire, histogramme fréquences,
    scatter dom_freq vs kurtosis, grille de waveforms de référence.
    Retourne le chemin HTML généré.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = signal_profile.get("events", [])
    signal_id = signal_profile.get("signal_id", "signal")
    n_clusters = signal_profile.get("n_clusters", 0)

    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for ev in events:
        by_cluster[ev["cluster_id"]].append(ev)

    colors = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
    ]

    def _color(cid: int) -> str:
        return colors[cid % len(colors)]

    html_parts: list[str] = [
        "<html><head><meta charset='utf-8'>",
        f"<title>Rapport Adapte2 — {signal_id}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f5f6fa}",
        "h1,h2{color:#2c3e50}",
        ".card{background:#fff;margin-bottom:24px;padding:20px;border-radius:8px;"
        "box-shadow:0 2px 6px rgba(0,0,0,.12)}",
        ".badge{display:inline-block;padding:2px 10px;border-radius:12px;"
        "font-size:12px;color:#fff;margin:2px}",
        "img{max-width:100%;border:1px solid #ddd;border-radius:4px;margin-top:8px}",
        "</style></head><body>",
        f"<h1>Rapport d'Analyse — {signal_id}</h1>",
    ]

    # ── Section 1 : Résumé texte ──────────────────────────────────────────────
    html_parts += [
        "<div class='card'>",
        "<h2>1. Résumé</h2>",
        generate_summary(signal_profile),
        "</div>",
    ]

    if not events:
        html_parts.append("</body></html>")
        path = output_dir / "rapport_enrichi.html"
        path.write_text("\n".join(html_parts), encoding="utf-8")
        return str(path)

    # ── Section 2 : Gantt timeline ────────────────────────────────────────────
    fig_gantt = go.Figure()
    for ev in events:
        cid = ev["cluster_id"]
        t_start = pd.Timestamp(ev["ref_timestamp"])
        dur_ms = max(ev["duration_s"] * 1000, 200)
        t_end = t_start + pd.Timedelta(milliseconds=dur_ms)
        fig_gantt.add_trace(
            go.Bar(
                x=[(t_end - t_start).total_seconds()],
                y=[ev["event_id"]],
                base=[t_start.isoformat()],
                orientation="h",
                marker_color=_color(cid),
                name=f"Type {cid}",
                showlegend=ev["event_id"] == (by_cluster[cid][0]["event_id"]),
                hovertemplate=(
                    f"<b>Évt {ev['event_id']}</b><br>"
                    f"Type {cid}<br>"
                    f"Début : {t_start.strftime('%H:%M:%S')}<br>"
                    f"Durée : {ev['duration_s']:.1f}s<br>"
                    f"Fenêtres : {ev['window_count']}<br>"
                    f"Fréq. réf : {ev['ref_dom_freq']:.1f} Hz"
                    "<extra></extra>"
                ),
            )
        )

    fig_gantt.update_layout(
        title="Timeline Gantt des événements",
        xaxis_title="Temps",
        yaxis_title="Événement #",
        barmode="overlay",
        height=max(300, len(events) * 18 + 100),
        xaxis=dict(type="date"),
    )

    html_parts += [
        "<div class='card'>",
        "<h2>2. Timeline des événements</h2>",
        _fig_to_html(fig_gantt),
        "</div>",
    ]

    # ── Section 3 : Heatmap horaire (type × heure) ────────────────────────────
    t_start_day = pd.Timestamp(signal_profile["t_start"]) if signal_profile.get("t_start") else None
    if t_start_day and n_clusters > 0:
        hours = list(range(24))
        z_data = np.zeros((n_clusters, 24), dtype=int)
        type_labels = [f"Type {c}" for c in range(n_clusters)]
        for ev in events:
            cid = ev["cluster_id"]
            for ts in ev.get("window_timestamps", [ev["ref_timestamp"]]):
                h = pd.Timestamp(ts).hour
                if 0 <= cid < n_clusters:
                    z_data[cid, h] += 1

        fig_heat = go.Figure(
            go.Heatmap(
                z=z_data,
                x=[f"{h:02d}h" for h in hours],
                y=type_labels,
                colorscale="YlOrRd",
                hovertemplate="Type %{y}<br>Heure : %{x}<br>Fenêtres : %{z}<extra></extra>",
            )
        )
        fig_heat.update_layout(
            title="Densité horaire des anomalies par type",
            xaxis_title="Heure de la journée",
            yaxis_title="Type",
            height=max(300, n_clusters * 40 + 100),
        )
        html_parts += [
            "<div class='card'>",
            "<h2>3. Heatmap horaire par type</h2>",
            _fig_to_html(fig_heat),
            "</div>",
        ]

    # ── Section 4 : Histogramme fréquences dominantes par type ────────────────
    fig_freq = go.Figure()
    for cid in sorted(by_cluster):
        freqs = [f for ev in by_cluster[cid] for f in ev.get("dom_freqs", [ev["ref_dom_freq"]])]
        fig_freq.add_trace(
            go.Histogram(
                x=freqs,
                name=f"Type {cid}",
                marker_color=_color(cid),
                opacity=0.75,
                nbinsx=30,
            )
        )
    fig_freq.update_layout(
        title="Distribution des fréquences dominantes par type",
        xaxis_title="Fréquence dominante (Hz)",
        yaxis_title="Nombre de fenêtres",
        barmode="overlay",
        height=350,
    )
    html_parts += [
        "<div class='card'>",
        "<h2>4. Histogramme des fréquences dominantes</h2>",
        _fig_to_html(fig_freq),
        "</div>",
    ]

    # ── Section 5 : Tableau détail événements ────────────────────────────────
    table_header = ["#", "Type", "Début", "Fenêtres", "Durée (s)", "Fréq. moy. (Hz)", "NCC moy."]
    table_rows = []
    for ev in events:
        mf = np.mean(ev["dom_freqs"]) if ev["dom_freqs"] else 0.0
        mn = np.mean(ev["ncc_vs_ref"]) if ev["ncc_vs_ref"] else 1.0
        ts = pd.Timestamp(ev["ref_timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        table_rows.append([
            ev["event_id"], f"Type {ev['cluster_id']}", ts,
            ev["window_count"], f"{ev['duration_s']:.1f}", f"{mf:.1f}", f"{mn:.4f}",
        ])

    fig_table = go.Figure(
        go.Table(
            header=dict(values=table_header, fill_color="#2c3e50", font_color="white"),
            cells=dict(
                values=list(zip(*table_rows)) if table_rows else [[] for _ in table_header],
                fill_color="white",
                align="left",
            ),
        )
    )
    fig_table.update_layout(title="Détail des événements", height=max(300, len(events) * 25 + 100))
    html_parts += [
        "<div class='card'>",
        "<h2>5. Détail des événements</h2>",
        _fig_to_html(fig_table),
        "</div>",
    ]

    # ── Section 6 : Waveforms de référence ───────────────────────────────────
    if waveform_images:
        html_parts += [
            "<div class='card'>",
            "<h2>6. Waveforms de référence</h2>",
            "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px'>",
        ]
        for ev in events:
            ev_id = ev["event_id"]
            if ev_id not in waveform_images:
                continue
            ts = pd.Timestamp(ev["ref_timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            html_parts.append(
                f"<div style='border:1px solid #ddd;border-radius:6px;padding:12px'>"
                f"<p><b>Événement {ev_id}</b> &bull; Type {ev['cluster_id']} &bull; "
                f"{ts} &bull; {ev['window_count']} fenêtres</p>"
                f"<img src='{waveform_images[ev_id]}' alt='Ref evt {ev_id}'/>"
                f"</div>"
            )
        html_parts += ["</div>", "</div>"]

    html_parts.append("</body></html>")

    path = output_dir / "rapport_enrichi.html"
    path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"--- Rapport enrichi : '{path}' ---")
    return str(path)


def generate_correlation_dashboard(
    profiles: list[dict],
    cooccurrence_df: pd.DataFrame,
    lead_lag_results: dict,
    shared_types: list[dict],
    output_dir: Path,
) -> str:
    """Rapport HTML de corrélation inter-signaux.

    Sections : chevauchement temporel, matrice co-occurrence, lead-lag, types partagés.
    Retourne le chemin HTML.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_parts: list[str] = [
        "<html><head><meta charset='utf-8'>",
        "<title>Rapport Corrélation Inter-Signaux — Adapte2</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f5f6fa}",
        "h1,h2,h3{color:#2c3e50}",
        ".card{background:#fff;margin-bottom:24px;padding:20px;border-radius:8px;"
        "box-shadow:0 2px 6px rgba(0,0,0,.12)}",
        ".pair-row{display:flex;gap:20px;flex-wrap:wrap;margin-top:12px}",
        ".pair-col{flex:1;min-width:240px;background:#f8f9fa;border-radius:6px;"
        "padding:12px;border:1px solid #e0e0e0}",
        "img{max-width:100%;border:1px solid #ddd;border-radius:4px;margin-top:6px}",
        ".no-img{color:#888;font-style:italic}",
        "</style></head><body>",
        "<h1>Rapport de Corrélation Inter-Signaux</h1>",
    ]

    # ── Section 1 : Chevauchement temporel ───────────────────────────────────
    if profiles:
        fig_overlap = go.Figure()
        all_ts_min: list[pd.Timestamp] = []
        all_ts_max: list[pd.Timestamp] = []

        for profile in profiles:
            all_ts = sorted(
                pd.Timestamp(t)
                for ev in profile["events"]
                for t in ev.get("window_timestamps", [ev["ref_timestamp"]])
            )
            if not all_ts:
                continue
            all_ts_min.append(all_ts[0])
            all_ts_max.append(all_ts[-1])

            counts_s = pd.Series(all_ts)
            counts = counts_s.dt.floor("1min").value_counts().sort_index()
            max_c = counts.max()
            norm = counts / max_c if max_c > 0 else counts

            fig_overlap.add_trace(
                go.Scatter(
                    x=norm.index.tolist(),
                    y=norm.values.tolist(),
                    mode="lines",
                    fill="tozeroy",
                    opacity=0.5,
                    name=profile["signal_id"],
                )
            )

        fig_overlap.update_layout(
            title="Chevauchement temporel des anomalies (normalisé)",
            xaxis_title="Date / Heure",
            yaxis_title="Densité normalisée",
            height=350,
        )
        html_parts += [
            "<div class='card'>",
            "<h2>1. Chevauchement temporel</h2>",
            _fig_to_html(fig_overlap),
            "</div>",
        ]

    # ── Section 2 : Matrice de co-occurrence ─────────────────────────────────
    if not cooccurrence_df.empty:
        fig_cooc = go.Figure(
            go.Heatmap(
                z=cooccurrence_df.values.tolist(),
                x=cooccurrence_df.columns.tolist(),
                y=cooccurrence_df.index.tolist(),
                colorscale="Blues",
                hovertemplate="%{y} → %{x}<br>Co-occurrences : %{z}<extra></extra>",
            )
        )
        fig_cooc.update_layout(
            title="Matrice de co-occurrence (A → B dans ±30s)",
            height=max(300, len(cooccurrence_df) * 50 + 100),
        )
        html_parts += [
            "<div class='card'>",
            "<h2>2. Matrice de co-occurrence</h2>",
            _fig_to_html(fig_cooc),
            "</div>",
        ]

    # ── Section 3 : Lead-lag plots ────────────────────────────────────────────
    if lead_lag_results:
        fig_ll = make_subplots(
            rows=1,
            cols=len(lead_lag_results),
            subplot_titles=list(lead_lag_results.keys()),
        )
        for col_idx, (pair_label, ll) in enumerate(lead_lag_results.items(), 1):
            fig_ll.add_trace(
                go.Bar(
                    x=ll["lags"],
                    y=ll["counts"],
                    name=pair_label,
                    marker_color="#3498db",
                ),
                row=1,
                col=col_idx,
            )
        fig_ll.update_layout(title="Distributions lead-lag entre paires de signaux", height=350)
        html_parts += [
            "<div class='card'>",
            "<h2>3. Lead-lag entre signaux</h2>",
            _fig_to_html(fig_ll),
            "</div>",
        ]

    # ── Section 4 : Types partagés ────────────────────────────────────────────
    html_parts += [
        "<div class='card'>",
        f"<h2>4. Types partagés ({len(shared_types)} paire(s) détectée(s))</h2>",
    ]
    if not shared_types:
        html_parts.append("<p style='color:#888;font-style:italic'>Aucun type partagé détecté.</p>")
    else:
        for idx, sp in enumerate(shared_types, 1):
            html_parts.append(
                f"<h3>Paire #{idx} — NCC = {sp['ncc']:.3f}</h3>"
                "<div class='pair-row'>"
            )
            for side in ("a", "b"):
                sig = sp[f"sig_{side}"]
                cid = sp[f"cid_{side}"]
                freq = sp[f"freq_{side}"]
                cnt = sp[f"count_{side}"]
                img = sp.get(f"img_{side}")
                img_html = (
                    f"<img src='{img}' alt='Type {cid} — {sig}'/>"
                    if img and os.path.exists(img)
                    else "<p class='no-img'>Image non disponible.</p>"
                )
                html_parts.append(
                    f"<div class='pair-col'>"
                    f"<h3>{sig} — Type {cid}</h3>"
                    f"<p>{freq:.0f} Hz &nbsp;·&nbsp; {cnt} occurrence(s)</p>"
                    f"{img_html}"
                    f"</div>"
                )
            html_parts.append("</div>")
    html_parts += ["</div>", "</body></html>"]

    path = output_dir / "rapport_correlation.html"
    path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"--- Rapport corrélation : '{path}' ---")
    return str(path)
