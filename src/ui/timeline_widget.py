"""Widget Tkinter : timeline Gantt des anomalies (canvas maison, pas matplotlib
pour rester réactif dans l'UI)."""

import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime as _dt, timedelta as _td

from src.reporting.palette import TYPE_COLORS


class TimelineWidget:
    """Canvas Gantt — une ligne par signal, barres colorées par type d'anomalie.
    Clic sur une barre → ouvre l'image de référence du type correspondant."""

    ROW_H      = 40
    LABEL_W    = 190
    LEGEND_W   = 130
    BAR_PAD    = 8
    MIN_BAR_PX = 4
    TL_W       = 680   # largeur de la zone temporelle en pixels

    def __init__(self, parent):
        self._canvas = tk.Canvas(parent, bg='white', height=70,
                                  highlightthickness=1, highlightbackground='#ccc')
        self._hbar = ttk.Scrollbar(parent, orient='horizontal', command=self._canvas.xview)
        self._vbar = ttk.Scrollbar(parent, orient='vertical',   command=self._canvas.yview)
        self._canvas.configure(xscrollcommand=self._hbar.set,
                                yscrollcommand=self._vbar.set)
        self._vbar.pack(side='right',  fill='y')
        self._hbar.pack(side='bottom', fill='x')
        self._canvas.pack(fill='both', expand=True)
        self._canvas.bind('<Button-1>', self._on_click)
        self._canvas.bind('<Motion>',   self._on_motion)
        self._bars: list[tuple] = []   # (x1, y1, x2, y2, img_path)
        self._canvas.create_text(
            14, 22, anchor='nw',
            text='Lancez une analyse pour visualiser la timeline des anomalies.',
            fill='#aaa', font=('Arial', 10, 'italic'),
        )

    def draw(self, entries: list[dict]):
        """
        entries : liste de dicts {
            'label'  : str   — "boitier_x / voie y"
            'events' : list  — chaque élément issu de signal_profile.json
                               + clé '_output_dir' injectée par _update_timeline
        }
        """
        self._canvas.delete('all')
        self._bars.clear()

        if not entries or not any(e['events'] for e in entries):
            self._canvas.configure(height=70)
            self._canvas.create_text(
                14, 22, anchor='nw',
                text='Aucune anomalie détectée pour la période sélectionnée.',
                fill='#aaa', font=('Arial', 10, 'italic'),
            )
            return

        # Plage temporelle globale
        all_ts = []
        for entry in entries:
            for ev in entry['events']:
                t = self._parse_ts(ev.get('ref_timestamp', ''))
                if t:
                    all_ts.append(t)
                    dur = max(ev.get('duration_s', 0), 0)
                    all_ts.append(t + _td(seconds=dur))
        if not all_ts:
            return

        t_min   = min(all_ts)
        t_max   = max(all_ts)
        total_s = max((t_max - t_min).total_seconds(), 1.0)

        n_rows   = len(entries)
        rows_h   = n_rows * self.ROW_H
        axis_h   = 34
        canvas_h = rows_h + axis_h
        canvas_w = self.LABEL_W + self.TL_W + self.LEGEND_W + 10

        self._canvas.configure(
            scrollregion=(0, 0, canvas_w, canvas_h),
            height=min(canvas_h + 4, 300),
        )

        all_types = sorted({ev['cluster_id']
                            for e in entries for ev in e['events']})

        for ri, entry in enumerate(entries):
            y0 = ri * self.ROW_H
            y1 = y0 + self.ROW_H
            bg = '#f7f7f7' if ri % 2 == 0 else '#ffffff'
            self._canvas.create_rectangle(0, y0, canvas_w, y1,
                                           fill=bg, outline='#e0e0e0')
            self._canvas.create_text(
                self.LABEL_W - 10, (y0 + y1) // 2,
                anchor='e', text=entry['label'],
                font=('Arial', 9), fill='#333',
            )
            self._canvas.create_line(self.LABEL_W, y0, self.LABEL_W, y1,
                                      fill='#bbb')

            for ev in entry['events']:
                cid   = ev.get('cluster_id', 0)
                color = TYPE_COLORS[cid % len(TYPE_COLORS)]
                t_ev  = self._parse_ts(ev.get('ref_timestamp', ''))
                if t_ev is None:
                    continue
                dur_s = max(ev.get('duration_s', 0), total_s * 0.003)
                off_s = (t_ev - t_min).total_seconds()

                bx1 = self.LABEL_W + int(off_s / total_s * self.TL_W)
                bx2 = self.LABEL_W + int((off_s + dur_s) / total_s * self.TL_W)
                bx2 = max(bx2, bx1 + self.MIN_BAR_PX)
                by1 = y0 + self.BAR_PAD
                by2 = y1 - self.BAR_PAD

                ev_id    = ev.get('event_id', 0)
                out_dir  = ev.get('_output_dir', '')
                img_path = os.path.join(out_dir, 'refs',
                                        f"event_{ev_id:02d}_type{cid}_ref.png")

                self._canvas.create_rectangle(bx1, by1, bx2, by2,
                                               fill=color, outline='')
                self._bars.append((bx1, by1, bx2, by2, img_path))

        # Axe X
        ax_y  = rows_h + 8
        ax_x0 = self.LABEL_W
        ax_x1 = self.LABEL_W + self.TL_W
        self._canvas.create_line(ax_x0, ax_y, ax_x1 + 12, ax_y,
                                  fill='#444', width=2)
        self._canvas.create_line(ax_x1 + 12, ax_y - 5,
                                  ax_x1 + 12, ax_y + 5, fill='#444', width=2)

        fmt = '%H:%M' if total_s < 86400 else '%d/%m %H:%M'
        for frac in (0, 0.25, 0.5, 0.75, 1.0):
            t_lbl = t_min + _td(seconds=total_s * frac)
            tx    = ax_x0 + int(frac * self.TL_W)
            self._canvas.create_line(tx, ax_y, tx, ax_y + 5, fill='#555')
            self._canvas.create_text(tx, ax_y + 7, anchor='n',
                text=t_lbl.strftime(fmt), font=('Arial', 8), fill='#555')

        self._canvas.create_text(
            ax_x0 + self.TL_W // 2, ax_y + 22, anchor='n',
            text="durée d'analyse →",
            font=('Arial', 9, 'italic'), fill='#666',
        )

        # Légende
        lx = self.LABEL_W + self.TL_W + 14
        ly = 10
        self._canvas.create_text(lx, ly, anchor='nw',
            text='Types :', font=('Arial', 9, 'bold'), fill='#333')
        ly += 18
        for cid in all_types:
            color = TYPE_COLORS[cid % len(TYPE_COLORS)]
            self._canvas.create_rectangle(lx, ly, lx + 22, ly + 14,
                                           fill=color, outline='')
            self._canvas.create_text(lx + 26, ly, anchor='nw',
                text=f'type {cid}', font=('Arial', 9), fill='#333')
            ly += 20

    @staticmethod
    def _parse_ts(s: str):
        """Parse une chaîne ISO en datetime, None si invalide."""
        if not s:
            return None
        try:
            return _dt.fromisoformat(s[:19])
        except ValueError:
            return None

    def _on_click(self, event):
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        for x1, y1, x2, y2, img_path in reversed(self._bars):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if os.path.exists(img_path):
                    subprocess.Popen(['xdg-open', img_path])
                else:
                    messagebox.showinfo('Image introuvable',
                                        f'Fichier attendu :\n{img_path}')
                return

    def _on_motion(self, event):
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        over = any(x1 <= cx <= x2 and y1 <= cy <= y2
                   for x1, y1, x2, y2, _ in self._bars)
        self._canvas.configure(cursor='hand2' if over else '')
