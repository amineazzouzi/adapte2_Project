"""Application Tkinter principale — pipeline d'analyse oscillo.

Point d'entrée unique du projet (lancé via interface.py à la racine).

Lance scripts/oscillo_analysis.py — un run par signal (cf.
src/analysis/oscillo_pipeline.py), mais tous les signaux EN PARALLÈLE via
src/analysis/batch_pipeline.OscilloBatchRunner (un GPU par process,
round-robin sur le nombre de GPUs détecté sur la machine) — puis
scripts/oscillo_correlation.py (cf. src/analysis/correlation_pipeline.py)
une fois que toutes les analyses sont terminées. Toujours de vrais arguments
CLI en subprocess (plus de patch du code source à la volée) — un pipeline
crashé ne peut pas casser la fenêtre, et la VRAM CUDA est proprement libérée
entre deux runs.
"""

import os
import re
import sys
import json
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import date as Date, timedelta

from src.core.config import BatchConfig
from src.core.paths import output_dir_for
from src.analysis.batch_pipeline import OscilloBatchRunner, detect_gpu_count
from src.ui.signal_row import SignalRow
from src.ui.timeline_widget import TimelineWidget

# ── Chemins ──────────────────────────────────────────────────────────────────
# SCRIPT_DIR : racine du projet (où vivent scripts/, results/, outputs_*/...).
# PROJECT_ROOT : un niveau au-dessus — c'est là qu'habite le .venv réel.
SCRIPT_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT    = os.path.dirname(SCRIPT_DIR)
VENV_PYTHON     = os.path.join(PROJECT_ROOT, '.venv', 'bin', 'python')
PYTHON_EXE      = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

# ── Couleurs log ──────────────────────────────────────────────────────────────
C_BG     = '#1e1e1e'
C_FG     = '#d4d4d4'
C_GREEN  = '#4ec94e'
C_YELLOW = '#dcdcaa'
C_RED    = '#f44747'
C_BLUE   = '#569cd6'
C_HEADER = '#2c3e50'
C_LINK   = '#4fc3f7'

_FILE_LINK_RE = re.compile(
    r"'(/[^'\n]+)'"                                             # chemin absolu entre apostrophes
    r"|"
    r"(?<!['\w])(/[^\s,;:'\"\n]+\.(?:png|svg|pdf|csv|xlsx|json|txt|html))"  # chemin absolu nu
)


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> Date:
    return Date.fromisoformat(s.strip())


def _date_range(start: Date, end: Date) -> list[str]:
    """Retourne la liste des dates (YYYY-MM-DD) de start à end inclus."""
    if end < start:
        raise ValueError(f"Date de fin ({end}) antérieure à la date de début ({start}).")
    days = []
    cur = start
    while cur <= end:
        days.append(cur.isoformat())
        cur += timedelta(days=1)
    return days


def _run_subprocess(argv, cwd, on_line, stop_flag):
    """Exécute argv en subprocess, diffuse chaque ligne de stdout à on_line."""
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        if stop_flag():
            proc.terminate()
            break
        on_line(line)
    proc.wait()
    return proc.returncode


# ─────────────────────────────────────────────────────────────────────────────
#  Application principale
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Pipeline Analyse Oscillo')
        self.geometry('1000x720')
        self.resizable(True, True)
        self.configure(bg='#f4f6f8')

        self._rows: list[SignalRow] = []
        self._running = False
        self._stop_requested = False
        self._link_counter = 0
        self._link_paths: dict[str, str] = {}

        self._build_ui()
        self._add_row()
        self._add_row()
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ── Construction de l'UI ─────────────────────────────────────────────────

    def _build_ui(self):
        # En-tête
        hdr = tk.Frame(self, bg=C_HEADER, pady=12)
        hdr.pack(fill='x')
        tk.Label(hdr, text='Pipeline Analyse Oscillo',
                 font=('Arial', 16, 'bold'), bg=C_HEADER, fg='white').pack()
        tk.Label(hdr, text='oscillo_analysis.py  →  oscillo_correlation.py',
                 font=('Arial', 9), bg=C_HEADER, fg='#aab7c4').pack()

        # Panneau signaux
        sig_lf = ttk.LabelFrame(self, text=' Signaux à analyser ', padding=6)
        sig_lf.pack(fill='x', padx=14, pady=10)

        # En-tête colonnes
        hdr_row = tk.Frame(sig_lf, bg='#f4f6f8')
        hdr_row.pack(fill='x', padx=6)
        for text, w in [('Boîtier', 12), ('Voie', 6), ('Date début', 11), ('Date fin', 11)]:
            tk.Label(hdr_row, text=text, font=('Arial', 9, 'bold'),
                     bg='#f4f6f8', fg='#555').pack(side='left', padx=6)

        self._sig_container = tk.Frame(sig_lf, bg='#f4f6f8')
        self._sig_container.pack(fill='x')

        ttk.Button(sig_lf, text='＋  Ajouter un signal',
                   command=self._add_row).pack(anchor='w', pady=(6, 2))

        # Résumé du plan (mis à jour dynamiquement)
        self._plan_var = tk.StringVar(value='')
        tk.Label(sig_lf, textvariable=self._plan_var, fg='#27ae60',
                 bg='#f4f6f8', font=('Arial', 9, 'italic')).pack(anchor='w', padx=6)

        # Barre de contrôle
        ctrl = tk.Frame(self, bg='#f4f6f8', pady=6)
        ctrl.pack(fill='x', padx=14)

        self._run_btn = ttk.Button(ctrl, text='▶   Lancer l\'analyse',
                                   command=self._start)
        self._run_btn.pack(side='left', padx=4)

        self._stop_btn = ttk.Button(ctrl, text='⬛  Arrêter',
                                    command=self._stop, state='disabled')
        self._stop_btn.pack(side='left', padx=4)

        self._open_btn = ttk.Button(ctrl, text='📂  Ouvrir les résultats',
                                    command=self._open_results, state='disabled')
        self._open_btn.pack(side='right', padx=4)

        self._corr_btn = ttk.Button(ctrl, text='📄  Rapport corrélation',
                                    command=self._open_corr_report, state='disabled')
        self._corr_btn.pack(side='right', padx=4)

        ttk.Button(ctrl, text='⏻  Quitter',
                   command=self._on_close).pack(side='right', padx=4)

        # Rapports enrichis (boutons dynamiques)
        self._enrichi_lf = ttk.LabelFrame(self, text=' Rapports enrichis ', padding=4)
        self._enrichi_lf.pack(fill='x', padx=14, pady=(0, 4))
        self._enrichi_inner = tk.Frame(self._enrichi_lf, bg='#f4f6f8')
        self._enrichi_inner.pack(fill='x')
        tk.Label(self._enrichi_lf, text='Aucune analyse effectuée.',
                 fg='#888', bg='#f4f6f8', font=('Arial', 9, 'italic'),
                 ).pack(anchor='w', padx=4)
        self._enrichi_placeholder = self._enrichi_lf.winfo_children()[-1]

        # Timeline des anomalies
        timeline_lf = ttk.LabelFrame(self, text=' Timeline des anomalies ', padding=4)
        timeline_lf.pack(fill='x', padx=14, pady=(0, 4))
        self._timeline = TimelineWidget(timeline_lf)

        # Barre de statut + progression
        self._status_var = tk.StringVar(value='En attente…')
        tk.Label(self, textvariable=self._status_var, anchor='w',
                 bg='#dfe6e9', font=('Arial', 10), relief='sunken',
                 padx=8).pack(fill='x', padx=14, pady=(0, 2))

        self._progress = ttk.Progressbar(self, mode='determinate')
        self._progress.pack(fill='x', padx=14, pady=(0, 6))

        # Journal
        log_lf = ttk.LabelFrame(self, text=' Journal d\'exécution ', padding=4)
        log_lf.pack(fill='both', expand=True, padx=14, pady=(0, 10))

        self._log = scrolledtext.ScrolledText(
            log_lf, wrap='word', font=('Courier New', 9),
            bg=C_BG, fg=C_FG, insertbackground='white',
        )
        self._log.pack(fill='both', expand=True)
        self._log.bind('<Key>', lambda e: 'break')
        self._log.bind('<Button-2>', lambda e: 'break')
        self._log.bind('<<Paste>>', lambda e: 'break')

        # Tags de couleur
        self._log.tag_config('green',    foreground=C_GREEN)
        self._log.tag_config('yellow',  foreground=C_YELLOW)
        self._log.tag_config('red',     foreground=C_RED)
        self._log.tag_config('blue',    foreground=C_BLUE)
        self._log.tag_config('header',  foreground='#c792ea',
                              font=('Courier New', 9, 'bold'))
        self._log.tag_config('filelink', foreground=C_LINK, underline=True)

        self._log.bind('<Button-1>', self._on_log_click)
        self._log.bind('<Motion>',   self._on_log_motion)

    # ── Gestion des lignes de signal ─────────────────────────────────────────

    def _add_row(self):
        row = SignalRow(self._sig_container, self._remove_row)
        # Mettre à jour le résumé quand l'utilisateur change les dates
        for var in (row.boitier_var, row.voie_var, row.date_start_var, row.date_end_var):
            var.trace_add('write', lambda *_: self._update_plan())
        self._rows.append(row)
        self._update_plan()

    def _remove_row(self, row: SignalRow):
        if len(self._rows) <= 1:
            messagebox.showwarning('Attention', 'Il faut garder au moins un signal.')
            return
        self._rows.remove(row)
        row.destroy()
        self._update_plan()

    def _update_plan(self):
        """Recalcule et affiche le nombre de signaux et de jours couverts."""
        total_days = 0
        signals = set()
        for row in self._rows:
            try:
                boitier, voie, ds, de = row.values()
                total_days += len(_date_range(_parse_date(ds), _parse_date(de)))
                signals.add((boitier, voie))
            except Exception:
                pass
        if total_days == 0:
            self._plan_var.set('')
        else:
            self._plan_var.set(
                f'→  {len(signals)} signal(s) à analyser '
                f'({total_days} jour(s) au total, combinés par signal)'
            )

    # ── Log helpers ──────────────────────────────────────────────────────────

    def _log_line(self, text: str):
        tag = None
        t = text.strip()
        if t.startswith('✅') or t.startswith('OK') or '✓' in t:
            tag = 'green'
        elif t.startswith('⚠') or 'WARNING' in t or 'AVERT' in t:
            tag = 'yellow'
        elif t.startswith('❌') or 'ERROR' in t or 'ERREUR' in t:
            tag = 'red'
        elif t.startswith('===') or t.startswith('---') or (t.startswith('[') and t.endswith(']')):
            tag = 'header'
        elif t.startswith('⏱') or '→' in t:
            tag = 'blue'

        last = 0
        for m in _FILE_LINK_RE.finditer(text):
            filepath = m.group(1) or m.group(2)
            if last < m.start():
                self._log.insert('end', text[last:m.start()], tag or '')

            self._link_counter += 1
            ltag = f'link_{self._link_counter}'
            self._log.tag_configure(ltag, foreground=C_LINK, underline=True)
            self._link_paths[ltag] = filepath

            tags = (tag, ltag) if tag else ltag
            self._log.insert('end', text[m.start():m.end()], tags)
            last = m.end()

        if last < len(text):
            self._log.insert('end', text[last:], tag or '')

        self._log.see('end')

    def _log_section(self, title: str):
        sep = '═' * 62
        self._log_line(f'\n{sep}\n  {title}\n{sep}\n')

    def _clear_log(self):
        self._log.delete('1.0', 'end')
        self._link_paths.clear()

    # ── Contrôles run/stop ───────────────────────────────────────────────────

    def _set_running(self, running: bool):
        self._running = running
        if not running:
            self._stop_requested = False
        self._run_btn.configure(state='disabled' if running else 'normal')
        self._stop_btn.configure(state='normal' if running else 'disabled')
        if running:
            self._open_btn.configure(state='disabled')
            self._corr_btn.configure(state='disabled')
        if running:
            self._progress.configure(mode='indeterminate')
            self._progress.start(10)
        else:
            self._progress.stop()
            self._progress.configure(mode='determinate', value=0)

    def _set_status(self, text: str):
        self._status_var.set(text)

    def _set_progress(self, value: float):
        """value entre 0 et 100."""
        self._progress.stop()
        self._progress.configure(mode='determinate', value=value)

    # ── Démarrage ────────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return

        # Valider les lignes et fusionner celles qui partagent le même signal
        # (boitier+voie) -> un seul job par signal, avec l'union de ses dates.
        jobs_by_signal: dict = {}  # (boitier, voie) -> {'boitier','voie','dates': set}
        for row in self._rows:
            try:
                boitier, voie, ds, de = row.values()
            except (ValueError, tk.TclError) as e:
                messagebox.showerror('Erreur de saisie', str(e))
                return
            if not boitier:
                messagebox.showerror('Erreur', 'Le nom du boîtier ne peut pas être vide.')
                return
            try:
                dates = _date_range(_parse_date(ds), _parse_date(de))
            except ValueError as e:
                messagebox.showerror('Erreur de date', str(e))
                return
            key = (boitier, voie)
            entry = jobs_by_signal.setdefault(
                key, {'boitier': boitier, 'voie': voie, 'dates': set()}
            )
            entry['dates'].update(dates)

        jobs = []
        for entry in jobs_by_signal.values():
            entry['dates'] = sorted(entry['dates'])
            jobs.append(entry)

        if not jobs:
            messagebox.showerror('Erreur', 'Aucun job à exécuter.')
            return

        self._clear_log()
        for w in self._enrichi_inner.winfo_children():
            w.destroy()
        self._enrichi_placeholder.pack(anchor='w', padx=4)
        self._set_running(True)
        self._set_status('Démarrage…')
        threading.Thread(target=self._pipeline, args=(jobs,), daemon=True).start()

    def _stop(self):
        self._stop_requested = True
        self._set_status('Arrêt demandé…')

    # ── Thread pipeline ───────────────────────────────────────────────────────

    def _pipeline(self, jobs: list[dict]):
        ok = True
        n  = len(jobs)
        completed = 0
        try:
            n_gpus = detect_gpu_count()
            batch_config = BatchConfig(
                jobs=jobs,
                log_dir=os.path.join(SCRIPT_DIR, 'results', 'batch_logs'),
                python_executable=PYTHON_EXE,
            )
            max_parallel = batch_config.max_parallel or n_gpus
            mode = 'en parallèle' if max_parallel > 1 else 'en série (1 GPU)'
            self.after(0, self._log_section,
                       f'DÉMARRAGE — {n} signal(s) à analyser '
                       f'({n_gpus} GPU(s) détecté(s), {mode})')

            def _on_line(label, line):
                self.after(0, self._log_line, f'[{label}] {line}')

            def _on_job_done(label, rc):
                nonlocal ok, completed
                completed += 1
                self.after(0, self._set_status,
                           f'[{completed}/{n}] terminé : {label}')
                self.after(0, self._set_progress, completed / n * 90)
                if rc not in (0, None):
                    ok = False
                    self.after(0, self._log_line,
                               f'\n⚠️  {label} terminé avec code {rc}\n')

            OscilloBatchRunner(batch_config, SCRIPT_DIR).run(
                on_line=_on_line,
                on_job_done=_on_job_done,
                stop_flag=lambda: self._stop_requested,
            )

            if self._stop_requested:
                self.after(0, self._set_status, '⬛ Arrêté.')
                self.after(0, self._log_line, '\n⬛ Pipeline interrompu par l\'utilisateur.\n')
                return

            self.after(0, self._set_progress, 95)

            if len(jobs) >= 2:
                self.after(0, self._set_status, 'Corrélation inter-signaux…')
                self.after(0, self._log_section,
                           f'CORRÉLATION — {len(jobs)} signal(s)')
                rc = self._run_correlation(jobs)
                if rc not in (0, None):
                    self.after(0, self._log_line,
                               f'\n⚠️  Corrélation terminée avec code {rc}\n')
                    ok = False
            else:
                self.after(0, self._log_line,
                           '\nℹ️  Corrélation ignorée (moins de 2 signaux).\n')

            self.after(0, self._set_progress, 100)

            self.after(0, self._populate_enrichi_buttons, jobs)
            self.after(0, self._update_timeline, jobs)

            if ok:
                self.after(0, self._set_status, '✅ Pipeline terminé avec succès.')
                self.after(0, self._log_line, '\n✅ Pipeline terminé avec succès.\n')
                self.after(0, lambda: self._open_btn.configure(state='normal'))
                self.after(0, lambda: self._corr_btn.configure(state='normal'))
            else:
                self.after(0, self._set_status, '⚠️  Terminé avec des avertissements.')

        except Exception as exc:
            self.after(0, self._log_line, f'\n❌ Exception : {exc}\n')
            self.after(0, self._set_status, f'❌ Erreur : {exc}')
        finally:
            self.after(0, self._set_running, False)

    # ── Lancement des scripts ─────────────────────────────────────────────────

    def _run_correlation(self, jobs: list[dict]) -> int:
        argv = [PYTHON_EXE, '-m', 'scripts.oscillo_correlation', '--project-dir', SCRIPT_DIR]
        for j in jobs:
            argv += ['--signal', j['boitier'], str(j['voie'])]
        return _run_subprocess(
            argv, SCRIPT_DIR,
            on_line=lambda l: self.after(0, self._log_line, l),
            stop_flag=lambda: self._stop_requested,
        )

    # ── Rapports enrichis dynamiques ─────────────────────────────────────────

    def _populate_enrichi_buttons(self, jobs: list[dict]):
        for w in self._enrichi_inner.winfo_children():
            w.destroy()
        self._enrichi_placeholder.pack_forget()

        for job in jobs:
            b, v, dates = job['boitier'], job['voie'], job['dates']
            date_label = dates[0] if len(dates) == 1 else f'{dates[0]} → {dates[-1]}'
            report = os.path.join(
                SCRIPT_DIR, output_dir_for(b, v), 'rapport_enrichi.html'
            )
            label = f'📄  {b} / voie_{v} / {date_label}'
            state = 'normal' if os.path.isfile(report) else 'disabled'
            ttk.Button(
                self._enrichi_inner, text=label, state=state,
                command=lambda p=report: subprocess.Popen(['xdg-open', p]),
            ).pack(side='left', padx=4, pady=2)

    # ── Clics sur les liens dans le journal ──────────────────────────────────

    def _on_log_click(self, event):
        idx = self._log.index(f'@{event.x},{event.y}')
        for tag in self._log.tag_names(idx):
            if tag in self._link_paths:
                path = self._link_paths[tag]
                if os.path.exists(path):
                    subprocess.Popen(['xdg-open', path])
                break

    def _on_log_motion(self, event):
        idx = self._log.index(f'@{event.x},{event.y}')
        over_link = any(t in self._link_paths for t in self._log.tag_names(idx))
        self._log.configure(cursor='hand2' if over_link else '')

    # ── Timeline des anomalies ────────────────────────────────────────────────

    def _update_timeline(self, jobs: list[dict]):
        """Lit le signal_profile.json de chaque signal et redessine le Gantt."""
        entries = []
        for job in jobs:
            out_dir = os.path.join(SCRIPT_DIR, output_dir_for(job['boitier'], job['voie']))
            profile_path = os.path.join(out_dir, 'signal_profile.json')
            events = []
            if os.path.isfile(profile_path):
                try:
                    with open(profile_path, encoding='utf-8') as fh:
                        profile = json.load(fh)
                    for ev in profile.get('events', []):
                        ev['_output_dir'] = out_dir
                        events.append(ev)
                except Exception:
                    pass
            if events:
                entries.append({
                    'label':  f"{job['boitier']} / voie {job['voie']}",
                    'events': events,
                })

        self._timeline.draw(entries)

    # ── Ouvrir les résultats ─────────────────────────────────────────────────

    def _open_results(self):
        output_dir = os.path.join(SCRIPT_DIR, 'results', 'outputs_correlation')
        if os.path.isdir(output_dir):
            subprocess.Popen(['xdg-open', output_dir])
        else:
            messagebox.showinfo('Résultats introuvables',
                                f'Dossier attendu :\n{output_dir}')

    def _open_corr_report(self):
        report = os.path.join(SCRIPT_DIR, 'results', 'outputs_correlation', 'rapport_correlation.html')
        if os.path.isfile(report):
            subprocess.Popen(['xdg-open', report])
        else:
            messagebox.showinfo('Rapport introuvable',
                                f'Fichier attendu :\n{report}')

    def _on_close(self):
        if self._running:
            if not messagebox.askyesno(
                'Analyse en cours',
                'Une analyse est en cours.\nVoulez-vous l\'arrêter et quitter ?',
            ):
                return
            self._stop_requested = True
        self.destroy()
