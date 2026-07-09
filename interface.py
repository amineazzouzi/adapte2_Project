#!/usr/bin/env python3
# coding: utf-8
"""
Interface graphique Tkinter pour le pipeline d'analyse oscillo.

Workflow :
  1. L'utilisateur configure N signaux (boitier, voie, date_début → date_fin).
  2. Clic "Lancer" → oscillo_analysis.py s'exécute UNE FOIS PAR SIGNAL
     (boitier+voie), en combinant toute sa plage de dates dans un seul
     rapport_enrichi.html (pas un run par jour).
  3. Puis oscillo_correlation.py s'exécute sur l'ensemble des signaux.

L'implémentation vit dans src/ui/app.py — ce script n'est qu'un lanceur.
"""

from src.ui.app import App

if __name__ == '__main__':
    app = App()
    app.mainloop()
