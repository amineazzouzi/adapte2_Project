"""Vue calendrier mensuel HTML des anomalies multi-jours."""

from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd


def generate_calendar(
    events_by_date: dict[str, dict],
    output_dir: Path,
    month: str | None = None,
) -> str:
    """Génère un calendrier HTML mensuel colorant chaque jour par densité d'anomalies.

    events_by_date : {'YYYY-MM-DD': signal_profile, ...}
    month : 'YYYY-MM' — si None, déduit depuis les dates disponibles.
    Retourne le chemin HTML.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not events_by_date:
        path = output_dir / "calendrier.html"
        path.write_text("<p>Aucune donnée disponible.</p>", encoding="utf-8")
        return str(path)

    # Déduire l'année/mois depuis les données si non fourni
    if month is None:
        month = sorted(events_by_date.keys())[0][:7]
    year, mon = int(month.split("-")[0]), int(month.split("-")[1])
    month_label = pd.Timestamp(f"{year}-{mon:02d}-01").strftime("%B %Y")

    # Calculer la densité par jour (nombre d'événements)
    counts_by_day: dict[int, int] = {}
    links_by_day: dict[int, str] = {}
    types_by_day: dict[int, set[int]] = {}

    for date_str, profile in events_by_date.items():
        try:
            day = int(date_str.split("-")[2])
        except (IndexError, ValueError):
            continue
        n_ev = len(profile.get("events", []))
        counts_by_day[day] = n_ev
        types_by_day[day] = {ev["cluster_id"] for ev in profile.get("events", [])}

        # Lien vers le rapport enrichi s'il existe
        report_path = output_dir.parent / f"outputs_{date_str}" / "rapport_enrichi.html"
        if report_path.exists():
            links_by_day[day] = str(report_path)

    max_count = max(counts_by_day.values(), default=1) or 1

    def _cell_color(day: int) -> str:
        n = counts_by_day.get(day, 0)
        if n == 0:
            return "#d5e8d4"  # vert clair
        ratio = min(n / max_count, 1.0)
        if ratio < 0.33:
            return "#ffe6cc"  # orange clair
        if ratio < 0.66:
            return "#ffcc99"  # orange
        return "#f8a4a4"     # rouge clair

    # Badges de couleur par type
    _type_colors = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e",
    ]

    def _type_badges(day: int) -> str:
        types = types_by_day.get(day, set())
        return " ".join(
            f"<span style='display:inline-block;padding:1px 6px;border-radius:10px;"
            f"background:{_type_colors[t % len(_type_colors)]};color:#fff;"
            f"font-size:10px'>T{t}</span>"
            for t in sorted(types)
        )

    cal = calendar.Calendar(firstweekday=0)  # lundi en premier
    weeks = cal.monthdayscalendar(year, mon)
    day_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    cells: list[str] = []
    for week in weeks:
        for day in week:
            if day == 0:
                cells.append("<div class='cell empty'></div>")
                continue
            color = _cell_color(day)
            n = counts_by_day.get(day, 0)
            badges = _type_badges(day)
            link = links_by_day.get(day)
            content = (
                f"<a href='file://{link}' style='text-decoration:none;color:inherit'>"
                if link else ""
            )
            content += (
                f"<div class='cell' style='background:{color}'>"
                f"<span class='day-num'>{day}</span>"
                f"<span class='ev-count'>{n} évt</span>"
                f"<div class='badges'>{badges}</div>"
                f"</div>"
            )
            if link:
                content += "</a>"
            cells.append(content)

    day_headers = "".join(f"<div class='day-header'>{d}</div>" for d in day_names)
    cells_html = "\n".join(cells)

    css = """
body{font-family:Arial,sans-serif;margin:24px;background:#f5f6fa}
h1{color:#2c3e50;text-align:center}
.calendar{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;max-width:900px;margin:0 auto}
.day-header{text-align:center;font-weight:bold;color:#555;padding:4px;font-size:12px}
.cell{min-height:80px;padding:6px;border-radius:6px;border:1px solid #ccc;cursor:pointer;
      display:flex;flex-direction:column;gap:2px}
.cell.empty{background:#f0f0f0;border:1px dashed #ccc}
.day-num{font-weight:bold;font-size:14px;color:#2c3e50}
.ev-count{font-size:11px;color:#555}
.badges{display:flex;flex-wrap:wrap;gap:2px}
.legend{display:flex;gap:20px;justify-content:center;margin:16px auto;max-width:900px}
.leg-item{display:flex;align-items:center;gap:6px;font-size:12px}
.leg-box{width:18px;height:18px;border-radius:4px;border:1px solid #ccc}
"""

    legend = """
<div class='legend'>
  <div class='leg-item'><div class='leg-box' style='background:#d5e8d4'></div> 0 événement</div>
  <div class='leg-item'><div class='leg-box' style='background:#ffe6cc'></div> Peu</div>
  <div class='leg-item'><div class='leg-box' style='background:#ffcc99'></div> Modéré</div>
  <div class='leg-item'><div class='leg-box' style='background:#f8a4a4'></div> Beaucoup</div>
</div>
"""

    html = "\n".join([
        "<html><head><meta charset='utf-8'>",
        f"<title>Calendrier — {month_label}</title>",
        f"<style>{css}</style></head><body>",
        f"<h1>Calendrier des anomalies — {month_label}</h1>",
        legend,
        "<div class='calendar'>",
        day_headers,
        cells_html,
        "</div>",
        "</body></html>",
    ])

    path = output_dir / f"calendrier_{month}.html"
    path.write_text(html, encoding="utf-8")
    print(f"--- Calendrier : '{path}' ---")
    return str(path)
