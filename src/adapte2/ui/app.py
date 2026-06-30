"""Application Tkinter principale — sans _patch_and_run."""

from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from datetime import date as Date, timedelta
from pathlib import Path
from subprocess import PIPE, STDOUT
from tkinter import messagebox, ttk

from .widgets import DataLakeBrowser, FileSystemBrowser, LogPanel, SignalRow

# Chemin absolu vers main.py (3 niveaux au-dessus de ce fichier : ui → adapte2 → src → projet)
MAIN_PY = Path(__file__).resolve().parents[4] / "main.py"
PYTHON_EXE = sys.executable

# Essai venv
_venv_py = MAIN_PY.parent / ".venv" / "bin" / "python"
if _venv_py.exists():
    PYTHON_EXE = str(_venv_py)

C_HEADER = "#2c3e50"


def _parse_date(s: str) -> Date:
    return Date.fromisoformat(s.strip())


def _date_range(start: Date, end: Date) -> list[str]:
    if end < start:
        raise ValueError(f"Date de fin ({end}) antérieure à la date de début ({start}).")
    days: list[str] = []
    cur = start
    while cur <= end:
        days.append(cur.isoformat())
        cur += timedelta(days=1)
    return days


class App(tk.Tk):
    """Interface principale du pipeline Adapte2."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Pipeline Analyse Oscillo — Adapte2")
        self.geometry("1100x780")
        self.resizable(True, True)
        self.configure(bg="#f4f6f8")

        self._rows: list[SignalRow] = []
        self._running = False
        self._stop_requested = False
        self._completed_jobs: list[dict] = []

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Construction UI ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # En-tête
        hdr = tk.Frame(self, bg=C_HEADER, pady=10)
        hdr.pack(fill="x")
        tk.Label(
            hdr, text="Pipeline Analyse Oscillo — Adapte2",
            font=("Arial", 15, "bold"), bg=C_HEADER, fg="white",
        ).pack()
        tk.Label(
            hdr, text="Analyse de signaux électriques agricoles par NCC GPU",
            font=("Arial", 9), bg=C_HEADER, fg="#aab7c4",
        ).pack()

        # Notebook (onglets)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=8)

        # Onglet 1 : sélection des signaux
        tab_sel = ttk.Frame(nb)
        nb.add(tab_sel, text="Sélection des signaux")
        self._build_tab_selection(tab_sel)

        # Onglet 2 : configuration
        tab_cfg = ttk.Frame(nb)
        nb.add(tab_cfg, text="Configuration")
        self._build_tab_config(tab_cfg)

        # Onglet 3 : journal
        tab_log = ttk.Frame(nb)
        nb.add(tab_log, text="Journal")
        self._build_tab_log(tab_log)

        # Barre de contrôle (hors notebook)
        ctrl = tk.Frame(self, bg="#f4f6f8", pady=6)
        ctrl.pack(fill="x", padx=12)

        self._run_btn = ttk.Button(ctrl, text="Lancer l'analyse", command=self._start)
        self._run_btn.pack(side="left", padx=4)

        self._stop_btn = ttk.Button(ctrl, text="Arreter", command=self._stop, state="disabled")
        self._stop_btn.pack(side="left", padx=4)

        ttk.Button(ctrl, text="Quitter", command=self._on_close).pack(side="right", padx=4)

        self._open_corr_btn = ttk.Button(
            ctrl, text="Rapport correlation", command=self._open_corr_report, state="disabled"
        )
        self._open_corr_btn.pack(side="right", padx=4)

        # Rapports par analyse
        self._enrichi_lf = ttk.LabelFrame(self, text=" Rapports enrichis ", padding=4)
        self._enrichi_lf.pack(fill="x", padx=12, pady=(0, 4))
        self._enrichi_inner = tk.Frame(self._enrichi_lf, bg="#f4f6f8")
        self._enrichi_inner.pack(fill="x")
        self._enrichi_placeholder = tk.Label(
            self._enrichi_lf, text="Aucune analyse effectuée.",
            fg="#888", bg="#f4f6f8", font=("Arial", 9, "italic"),
        )
        self._enrichi_placeholder.pack(anchor="w", padx=4)

        # Barre de statut + progression
        self._status_var = tk.StringVar(value="En attente…")
        tk.Label(
            self, textvariable=self._status_var,
            anchor="w", bg="#dfe6e9", font=("Arial", 10), relief="sunken", padx=8,
        ).pack(fill="x", padx=12, pady=(0, 2))

        self._progress = ttk.Progressbar(self, mode="determinate")
        self._progress.pack(fill="x", padx=12, pady=(0, 6))

    def _build_tab_selection(self, parent: ttk.Frame) -> None:
        sel_nb = ttk.Notebook(parent)
        sel_nb.pack(fill="both", expand=True, padx=4, pady=4)

        # Sous-onglet : Data Lake Browser
        tab_dl = ttk.Frame(sel_nb)
        sel_nb.add(tab_dl, text="Data Lake")
        from ..core.config import PathConfig
        dl_path = PathConfig().data_lake
        self._dl_browser = DataLakeBrowser(tab_dl, data_lake_path=dl_path)
        self._dl_browser.pack(fill="both", expand=True, padx=4, pady=4)
        tk.Label(
            tab_dl,
            text="Ctrl+clic pour multi-sélection. Double-cliquez sur une date pour sélectionner.",
            fg="#888", font=("Arial", 8),
        ).pack(anchor="w", padx=4)

        # Sous-onglet : Fichiers bruts
        tab_files = ttk.Frame(sel_nb)
        sel_nb.add(tab_files, text="Fichiers bruts")
        self._fs_browser = FileSystemBrowser(tab_files)
        self._fs_browser.pack(fill="both", expand=True, padx=4, pady=4)

        # Sous-onglet : Saisie manuelle (ancienne interface)
        tab_manual = ttk.Frame(sel_nb)
        sel_nb.add(tab_manual, text="Saisie manuelle")
        self._build_manual_tab(tab_manual)

    def _build_manual_tab(self, parent: ttk.Frame) -> None:
        hdr_row = tk.Frame(parent)
        hdr_row.pack(fill="x", padx=6, pady=2)
        for text in ["Boitier", "Voie", "Date debut", "Date fin"]:
            tk.Label(hdr_row, text=text, font=("Arial", 9, "bold"), fg="#555").pack(
                side="left", padx=8
            )

        self._sig_container = tk.Frame(parent)
        self._sig_container.pack(fill="x")

        ttk.Button(parent, text="+ Ajouter un signal", command=self._add_row).pack(
            anchor="w", pady=(6, 2)
        )
        self._plan_var = tk.StringVar(value="")
        tk.Label(
            parent, textvariable=self._plan_var,
            fg="#27ae60", font=("Arial", 9, "italic"),
        ).pack(anchor="w", padx=6)

        # Ajouter deux lignes par défaut
        self._add_row()
        self._add_row()

    def _build_tab_config(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text=" Seuils d'analyse ", padding=10)
        lf.pack(fill="x", padx=12, pady=12)

        params = [
            ("NCC threshold (groupement)", "ncc_threshold_var", "0.30"),
            ("NCC type threshold (clustering)", "ncc_type_var", "0.30"),
            ("NCC max lag (pts)", "ncc_max_lag_var", "1000"),
            ("Seuil amplitude", "amp_threshold_var", "50"),
            ("Nombre de segments", "n_seg_var", "10"),
            ("Workers CPU", "workers_var", "4"),
            ("GPU batch size", "gpu_batch_var", "2048"),
        ]
        for label, attr, default in params:
            row = tk.Frame(lf)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=35, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            setattr(self, f"_{attr}", var)
            ttk.Entry(row, textvariable=var, width=10).pack(side="left")

    def _build_tab_log(self, parent: ttk.Frame) -> None:
        self._log = LogPanel(parent)
        self._log.pack(fill="both", expand=True, padx=4, pady=4)

    # ── Gestion lignes saisie manuelle ───────────────────────────────────────

    def _add_row(self) -> None:
        row = SignalRow(self._sig_container, self._remove_row)
        row.pack(fill="x", padx=6, pady=3)
        for var in (row.boitier_var, row.voie_var, row.date_start_var, row.date_end_var):
            var.trace_add("write", lambda *_: self._update_plan())
        self._rows.append(row)
        self._update_plan()

    def _remove_row(self, row: SignalRow) -> None:
        if len(self._rows) <= 1:
            messagebox.showwarning("Attention", "Il faut garder au moins un signal.")
            return
        self._rows.remove(row)
        row.destroy()
        self._update_plan()

    def _update_plan(self) -> None:
        total = 0
        for row in self._rows:
            try:
                _, _, ds, de = row.values()
                total += len(_date_range(_parse_date(ds), _parse_date(de)))
            except Exception:
                pass
        self._plan_var.set(
            f"-> {total} analyse(s) au total" if total else ""
        )

    # ── Contrôles run/stop ───────────────────────────────────────────────────

    def _set_running(self, running: bool) -> None:
        self._running = running
        if not running:
            self._stop_requested = False
        self._run_btn.configure(state="disabled" if running else "normal")
        self._stop_btn.configure(state="normal" if running else "disabled")
        if running:
            self._progress.configure(mode="indeterminate")
            self._progress.start(10)
        else:
            self._progress.stop()
            self._progress.configure(mode="determinate", value=0)

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    def _set_progress(self, value: float) -> None:
        self._progress.stop()
        self._progress.configure(mode="determinate", value=value)

    # ── Démarrage ────────────────────────────────────────────────────────────

    def _start(self) -> None:
        if self._running:
            return

        jobs: list[dict] = []

        # 1. Sélection data lake browser
        dl_sel = self._dl_browser.get_selected()
        for item in dl_sel:
            jobs.append(item)

        # 2. Saisie manuelle
        for row in self._rows:
            try:
                boitier, voie, ds, de = row.values()
                if not boitier:
                    continue
                for d in _date_range(_parse_date(ds), _parse_date(de)):
                    job = {"boitier": boitier, "voie": voie, "date": d}
                    if job not in jobs:
                        jobs.append(job)
            except Exception:
                pass

        if not jobs:
            messagebox.showerror("Erreur", "Aucun signal/date sélectionné.")
            return

        self._completed_jobs.clear()
        self._log.clear()
        for w in self._enrichi_inner.winfo_children():
            w.destroy()
        self._enrichi_placeholder.pack(anchor="w", padx=4)
        self._set_running(True)
        self._set_status("Démarrage…")
        threading.Thread(target=self._pipeline, args=(jobs,), daemon=True).start()

    def _stop(self) -> None:
        self._stop_requested = True
        self._set_status("Arret demande…")

    # ── Thread pipeline ───────────────────────────────────────────────────────

    def _pipeline(self, jobs: list[dict]) -> None:
        n = len(jobs)
        ok = True
        try:
            self.after(0, self._log.append, f"\n=== DEMARRAGE — {n} analyse(s) ===\n")
            for i, job in enumerate(jobs, 1):
                if self._stop_requested:
                    break
                boitier, voie, date = job["boitier"], job["voie"], job["date"]
                label = f"{boitier} / voie_{voie} / {date}"
                self.after(0, self._set_status, f"[{i}/{n}] Analyse : {label}…")
                self.after(0, self._set_progress, (i - 1) / n * 100)
                self.after(0, self._log.append, f"\n=== ANALYSE ({i}/{n}) — {label} ===\n")

                rc = self._run_analysis(boitier, voie, date)
                if rc not in (0, None):
                    self.after(0, self._log.append, f"\n⚠ Analyse terminée avec code {rc}\n")
                    ok = False
                else:
                    self._completed_jobs.append(job)

            if self._stop_requested:
                self.after(0, self._set_status, "Arrete.")
                self.after(0, self._log.append, "\nPipeline interrompu par l'utilisateur.\n")
                return

            self.after(0, self._set_progress, 95)

            if len(self._completed_jobs) >= 2:
                self.after(0, self._set_status, "Correlation inter-signaux…")
                self.after(0, self._log.append, "\n=== CORRELATION ===\n")
                rc = self._run_correlation(self._completed_jobs)
                if rc not in (0, None):
                    self.after(0, self._log.append, f"\n⚠ Correlation terminée avec code {rc}\n")
                    ok = False
            else:
                self.after(0, self._log.append, "\nCorrelation ignorée (< 2 analyses).\n")

            self.after(0, self._set_progress, 100)
            self.after(0, self._populate_enrichi_buttons, self._completed_jobs)

            if ok:
                self.after(0, self._set_status, "OK Pipeline termine avec succes.")
                self.after(0, self._log.append, "\nOK Pipeline termine avec succes.\n")
                self.after(0, lambda: self._open_corr_btn.configure(state="normal"))
            else:
                self.after(0, self._set_status, "Termine avec des avertissements.")
        except Exception as exc:
            self.after(0, self._log.append, f"\nERREUR Exception : {exc}\n")
            self.after(0, self._set_status, f"Erreur : {exc}")
        finally:
            self.after(0, self._set_running, False)

    # ── Lancement subprocess main.py ─────────────────────────────────────────

    def _run_analysis(self, boitier: str, voie: int, date: str) -> int | None:
        cmd = [
            PYTHON_EXE, str(MAIN_PY), "analyze",
            "--boitier", boitier,
            "--voie", str(voie),
            "--date", date,
            "--output", f"outputs_{boitier}_v{voie}_{date}",
        ]
        return self._stream_subprocess(cmd)

    def _run_correlation(self, jobs: list[dict]) -> int | None:
        # Construit la liste des dossiers de profils
        profile_dirs = [
            f"outputs_{j['boitier']}_v{j['voie']}_{j['date']}"
            for j in jobs
        ]
        cmd = [PYTHON_EXE, str(MAIN_PY), "correlate", "--profiles"] + profile_dirs
        return self._stream_subprocess(cmd)

    def _stream_subprocess(self, cmd: list[str]) -> int | None:
        """Lance un subprocess et streame stdout vers le LogPanel."""
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=PIPE,
                stderr=STDOUT,
                text=True,
                bufsize=1,
                cwd=str(MAIN_PY.parent),
            )
            for line in proc.stdout:  # type: ignore[union-attr]
                if self._stop_requested:
                    proc.terminate()
                    break
                self.after(0, self._log.append, line)
            proc.wait()
            return proc.returncode
        except FileNotFoundError as exc:
            self.after(0, self._log.append, f"\nERREUR commande introuvable : {exc}\n")
            return 1

    # ── Rapports enrichis ────────────────────────────────────────────────────

    def _populate_enrichi_buttons(self, jobs: list[dict]) -> None:
        for w in self._enrichi_inner.winfo_children():
            w.destroy()
        self._enrichi_placeholder.pack_forget()

        for job in jobs:
            b, v, d = job["boitier"], job["voie"], job["date"]
            report = MAIN_PY.parent / f"outputs_{b}_v{v}_{d}" / "rapport_enrichi.html"
            label = f"{b} / voie_{v} / {d}"
            state = "normal" if report.is_file() else "disabled"
            ttk.Button(
                self._enrichi_inner,
                text=label,
                state=state,
                command=lambda p=str(report): subprocess.Popen(["xdg-open", p]),
            ).pack(side="left", padx=4, pady=2)

    def _open_corr_report(self) -> None:
        report = MAIN_PY.parent / "outputs_correlation" / "rapport_correlation.html"
        if report.is_file():
            subprocess.Popen(["xdg-open", str(report)])
        else:
            messagebox.showinfo("Rapport introuvable", f"Fichier attendu :\n{report}")

    # ── Fermeture ────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self._running:
            if not messagebox.askyesno(
                "Analyse en cours",
                "Une analyse est en cours.\nVoulez-vous l'arrêter et quitter ?",
            ):
                return
            self._stop_requested = True
        self.destroy()
