"""Génération de résumés texte automatiques à partir d'un signal_profile."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def generate_summary(signal_profile: dict) -> str:
    """Génère un paragraphe HTML descriptif de la journée d'analyse.

    Template-based, sans LLM. Retourne une chaîne HTML.
    """
    events = signal_profile.get("events", [])
    signal_id = signal_profile.get("signal_id", "inconnu")
    n_clusters = signal_profile.get("n_clusters", 0)
    t_start = signal_profile.get("t_start")
    date_str = ""
    if t_start:
        date_str = pd.Timestamp(t_start).strftime("%d/%m/%Y")

    if not events:
        return (
            f"<p>Le <b>{date_str}</b> : aucune anomalie détectée sur "
            f"<b>{signal_id}</b>.</p>"
        )

    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for ev in events:
        by_cluster[ev["cluster_id"]].append(ev)

    parts: list[str] = [
        f"<p>Le <b>{date_str}</b> : <b>{n_clusters}</b> type(s) d'anomalie(s) "
        f"détecté(s) sur <b>{signal_id}</b> "
        f"({len(events)} événements, {signal_profile.get('total_windows', 0)} fenêtres).</p>"
    ]

    # Description par type, trié par nombre d'occurrences décroissant
    for cid in sorted(by_cluster, key=lambda c: len(by_cluster[c]), reverse=True):
        c_events = by_cluster[cid]
        n_ev = len(c_events)
        all_freqs = [f for ev in c_events for f in ev.get("dom_freqs", [ev["ref_dom_freq"]])]
        mean_freq = sum(all_freqs) / len(all_freqs) if all_freqs else 0.0
        total_win = sum(ev["window_count"] for ev in c_events)

        ts_list = sorted(ev["ref_timestamp"] for ev in c_events)
        h1 = pd.Timestamp(ts_list[0]).strftime("%H:%M")
        h2 = pd.Timestamp(ts_list[-1]).strftime("%H:%M")

        parts.append(
            f"<p>Le <b>Type {cid}</b> ({mean_freq:.0f}&nbsp;Hz, "
            f"{n_ev} événement(s), {total_win} fenêtres) "
            f"est apparu entre <b>{h1}</b> et <b>{h2}</b>.</p>"
        )

    # Type dominant
    dominant_cid = max(by_cluster, key=lambda c: len(by_cluster[c]))
    if n_clusters > 1:
        n_dom = len(by_cluster[dominant_cid])
        parts.append(
            f"<p>Le type le plus fréquent est <b>Type {dominant_cid}</b> "
            f"avec {n_dom} événement(s).</p>"
        )

    return "\n".join(parts)
