"""Widget Tkinter : une ligne de signal (boitier, voie, plage de dates)."""

import tkinter as tk
from tkinter import ttk


class SignalRow:
    def __init__(self, parent, on_remove):
        self.frame = tk.Frame(parent, bg='#ecf0f1', relief='ridge', bd=1)
        self.frame.pack(fill='x', padx=6, pady=3)

        self.boitier_var    = tk.StringVar(value='boitier_1')
        self.voie_var       = tk.StringVar(value='1')
        self.date_start_var = tk.StringVar(value='2026-03-17')
        self.date_end_var   = tk.StringVar(value='2026-03-17')

        # Boîtier
        tk.Label(self.frame, text='Boîtier :', bg='#ecf0f1', anchor='e').pack(side='left', padx=(8, 2))
        ttk.Entry(self.frame, textvariable=self.boitier_var, width=12).pack(side='left', padx=2)

        # Voie
        tk.Label(self.frame, text='Voie :', bg='#ecf0f1', anchor='e').pack(side='left', padx=(8, 2))
        ttk.Spinbox(self.frame, from_=1, to=16, textvariable=self.voie_var,
                    width=4, wrap=True).pack(side='left', padx=2)

        # Séparateur
        ttk.Separator(self.frame, orient='vertical').pack(side='left', fill='y', padx=10, pady=4)

        # Date début
        tk.Label(self.frame, text='Du :', bg='#ecf0f1', anchor='e').pack(side='left', padx=(0, 2))
        ttk.Entry(self.frame, textvariable=self.date_start_var, width=11).pack(side='left', padx=2)

        # Date fin
        tk.Label(self.frame, text='Au :', bg='#ecf0f1', anchor='e').pack(side='left', padx=(6, 2))
        ttk.Entry(self.frame, textvariable=self.date_end_var, width=11).pack(side='left', padx=2)

        tk.Label(self.frame, text='(YYYY-MM-DD)', bg='#ecf0f1',
                 font=('Arial', 8), fg='#888').pack(side='left', padx=(2, 4))

        # Supprimer
        ttk.Button(self.frame, text='✕', width=3,
                   command=lambda: on_remove(self)).pack(side='right', padx=8)

    def values(self) -> tuple:
        """Retourne (boitier, voie, date_start_str, date_end_str)."""
        return (
            self.boitier_var.get().strip(),
            int(self.voie_var.get()),
            self.date_start_var.get().strip(),
            self.date_end_var.get().strip(),
        )

    def destroy(self):
        self.frame.destroy()
