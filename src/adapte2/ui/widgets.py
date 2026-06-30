"""Widgets Tkinter réutilisables pour l'interface Adapte2."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk
from typing import Callable


class DataLakeBrowser(ttk.Frame):
    """Arbre (Treeview) affichant la structure du data lake : boîtier → voie → date."""

    def __init__(self, parent: tk.Widget, data_lake_path: str, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._data_lake_path = Path(data_lake_path)
        self._build()
        self.refresh()

    def _build(self) -> None:
        scroll_y = ttk.Scrollbar(self, orient="vertical")
        scroll_y.pack(side="right", fill="y")

        self._tree = ttk.Treeview(
            self,
            yscrollcommand=scroll_y.set,
            selectmode="extended",
            columns=("type",),
        )
        self._tree.heading("#0", text="Données disponibles")
        self._tree.heading("type", text="Type")
        self._tree.column("type", width=80, anchor="center")
        self._tree.pack(fill="both", expand=True)
        scroll_y.config(command=self._tree.yview)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Actualiser", command=self.refresh).pack(side="left", padx=2)

    def refresh(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)
        if not self._data_lake_path.exists():
            return
        for boitier_dir in sorted(self._data_lake_path.iterdir()):
            if not boitier_dir.is_dir():
                continue
            b_node = self._tree.insert("", "end", text=boitier_dir.name, open=False)
            oscillo_dir = boitier_dir / "oscillo"
            if oscillo_dir.exists():
                for voie_dir in sorted(oscillo_dir.iterdir()):
                    if not voie_dir.is_dir():
                        continue
                    v_node = self._tree.insert(b_node, "end", text=voie_dir.name, values=("oscillo",))
                    for year_d in sorted(voie_dir.glob("year=*")):
                        year = year_d.name.split("=")[1]
                        for month_d in sorted(year_d.glob("month=*")):
                            month = month_d.name.split("=")[1]
                            for day_d in sorted(month_d.glob("day=*")):
                                if (day_d / "data.parquet").exists():
                                    day = day_d.name.split("=")[1]
                                    date_str = f"{year}-{int(month):02d}-{int(day):02d}"
                                    iid = f"{boitier_dir.name}|{voie_dir.name}|{date_str}"
                                    self._tree.insert(
                                        v_node, "end",
                                        iid=iid,
                                        text=date_str,
                                        values=("parquet",),
                                    )

    def get_selected(self) -> list[dict]:
        """Retourne la liste des items sélectionnés avec {boitier, voie, date}."""
        result = []
        for iid in self._tree.selection():
            parts = iid.split("|")
            if len(parts) == 3:
                boitier, voie_str, date = parts
                try:
                    voie = int(voie_str.split("_")[1])
                except (IndexError, ValueError):
                    continue
                result.append({"boitier": boitier, "voie": voie, "date": date})
        return result


class FileSystemBrowser(ttk.Frame):
    """Sélecteur de fichiers .txt bruts via une boîte de dialogue."""

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._files: list[Path] = []
        self._build()

    def _build(self) -> None:
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Parcourir…", command=self._browse).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Effacer tout", command=self._clear).pack(side="left", padx=2)

        scroll = ttk.Scrollbar(self, orient="vertical")
        scroll.pack(side="right", fill="y")
        self._listbox = tk.Listbox(
            self,
            selectmode="extended",
            yscrollcommand=scroll.set,
            height=8,
        )
        self._listbox.pack(fill="both", expand=True)
        scroll.config(command=self._listbox.yview)

        ttk.Button(self, text="Supprimer sélection", command=self._remove_selected).pack(
            anchor="w", padx=2, pady=2
        )

    def _browse(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Sélectionner des fichiers oscilloscope",
            filetypes=[("Fichiers texte / CSV", "*.txt *.csv"), ("Tous", "*.*")],
        )
        for p in paths:
            fp = Path(p)
            if fp not in self._files:
                self._files.append(fp)
                self._listbox.insert("end", str(fp))

    def _remove_selected(self) -> None:
        for idx in sorted(self._listbox.curselection(), reverse=True):
            self._listbox.delete(idx)
            del self._files[idx]

    def _clear(self) -> None:
        self._files.clear()
        self._listbox.delete(0, "end")

    def get_files(self) -> list[Path]:
        return list(self._files)


class SignalRow(ttk.Frame):
    """Ligne de saisie manuelle boîtier / voie / plage de dates."""

    def __init__(self, parent: tk.Widget, on_remove: Callable, **kwargs) -> None:
        super().__init__(parent, relief="ridge", **kwargs)
        self.boitier_var = tk.StringVar(value="boitier_1")
        self.voie_var = tk.StringVar(value="1")
        self.date_start_var = tk.StringVar(value="2026-03-17")
        self.date_end_var = tk.StringVar(value="2026-03-17")

        tk.Label(self, text="Boîtier :").pack(side="left", padx=(8, 2))
        ttk.Entry(self, textvariable=self.boitier_var, width=12).pack(side="left", padx=2)

        tk.Label(self, text="Voie :").pack(side="left", padx=(8, 2))
        ttk.Spinbox(self, from_=1, to=16, textvariable=self.voie_var, width=4, wrap=True).pack(
            side="left", padx=2
        )

        ttk.Separator(self, orient="vertical").pack(side="left", fill="y", padx=8, pady=4)

        tk.Label(self, text="Du :").pack(side="left", padx=(0, 2))
        ttk.Entry(self, textvariable=self.date_start_var, width=11).pack(side="left", padx=2)

        tk.Label(self, text="Au :").pack(side="left", padx=(6, 2))
        ttk.Entry(self, textvariable=self.date_end_var, width=11).pack(side="left", padx=2)

        tk.Label(self, text="(YYYY-MM-DD)", font=("Arial", 8), fg="#888").pack(
            side="left", padx=(2, 4)
        )

        ttk.Button(self, text="✕", width=3, command=lambda: on_remove(self)).pack(
            side="right", padx=8
        )

    def values(self) -> tuple[str, int, str, str]:
        return (
            self.boitier_var.get().strip(),
            int(self.voie_var.get()),
            self.date_start_var.get().strip(),
            self.date_end_var.get().strip(),
        )


class LogPanel(ttk.Frame):
    """Panneau de journal avec coloration syntaxique et liens cliquables."""

    _FILE_LINK_PAT = (
        r"'(/[^'\n]+)'"
        r"|"
        r"(?<!['\w])(/[^\s,;:'\"\n]+\.(?:png|svg|pdf|csv|xlsx|json|txt|html))"
    )

    # Couleurs
    C_BG = "#1e1e1e"
    C_FG = "#d4d4d4"
    C_GREEN = "#4ec94e"
    C_YELLOW = "#dcdcaa"
    C_RED = "#f44747"
    C_BLUE = "#569cd6"
    C_LINK = "#4fc3f7"

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        import re

        self._re = re.compile(self._FILE_LINK_PAT)
        self._link_counter = 0
        self._link_paths: dict[str, str] = {}
        self._build()

    def _build(self) -> None:
        import subprocess

        self._text = scrolledtext.ScrolledText(
            self,
            wrap="word",
            font=("Courier New", 9),
            bg=self.C_BG,
            fg=self.C_FG,
            insertbackground="white",
        )
        self._text.pack(fill="both", expand=True)
        self._text.bind("<Key>", lambda e: "break")

        for tag, color, bold in [
            ("green", self.C_GREEN, False),
            ("yellow", self.C_YELLOW, False),
            ("red", self.C_RED, False),
            ("blue", self.C_BLUE, False),
            ("header", "#c792ea", True),
        ]:
            font = ("Courier New", 9, "bold") if bold else ("Courier New", 9)
            self._text.tag_config(tag, foreground=color, font=font)

        def _on_click(event: tk.Event) -> None:  # type: ignore[name-defined]
            idx = self._text.index(f"@{event.x},{event.y}")
            for tag in self._text.tag_names(idx):
                if tag in self._link_paths:
                    path = self._link_paths[tag]
                    import os
                    if os.path.exists(path):
                        subprocess.Popen(["xdg-open", path])
                    break

        def _on_motion(event: tk.Event) -> None:  # type: ignore[name-defined]
            idx = self._text.index(f"@{event.x},{event.y}")
            over = any(t in self._link_paths for t in self._text.tag_names(idx))
            self._text.configure(cursor="hand2" if over else "")

        self._text.bind("<Button-1>", _on_click)
        self._text.bind("<Motion>", _on_motion)

    def append(self, text: str) -> None:
        """Ajoute une ligne avec coloration et détection de liens fichier."""
        tag = self._classify_tag(text)
        last = 0
        for m in self._re.finditer(text):
            filepath = m.group(1) or m.group(2)
            if last < m.start():
                self._text.insert("end", text[last : m.start()], tag or "")
            self._link_counter += 1
            ltag = f"link_{self._link_counter}"
            self._text.tag_configure(ltag, foreground=self.C_LINK, underline=True)
            self._link_paths[ltag] = filepath
            tags = (tag, ltag) if tag else ltag
            self._text.insert("end", text[m.start() : m.end()], tags)
            last = m.end()
        if last < len(text):
            self._text.insert("end", text[last:], tag or "")
        self._text.see("end")

    def clear(self) -> None:
        self._text.delete("1.0", "end")
        self._link_paths.clear()

    @staticmethod
    def _classify_tag(text: str) -> str | None:
        t = text.strip()
        if t.startswith("OK") or "OK" in t or t.startswith("✅"):
            return "green"
        if t.startswith("⚠") or "WARNING" in t or "AVERT" in t:
            return "yellow"
        if t.startswith("❌") or "ERROR" in t or "ERREUR" in t:
            return "red"
        if t.startswith("===") or t.startswith("---") or (t.startswith("[") and t.endswith("]")):
            return "header"
        if t.startswith("⏱") or "→" in t:
            return "blue"
        return None
