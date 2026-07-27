#!/usr/bin/env python
# coding: utf-8
"""
Worker CLI interne — pipeline de corrélation inter-signaux.

PAS un point d'entrée utilisateur : invoqué en subprocess par src/ui/app.py
une fois que tous les jobs oscillo (voir scripts/oscillo_analysis.py) sont
terminés — interface.py est le seul point d'entrée du projet.

Lit les signal_profile.json produits par scripts/oscillo_analysis.py (un
dossier par signal boitier+voie, cf. src/core/paths.py) et détecte les types
d'anomalies partagés entre les signaux sélectionnés.

Invocation : `python -m scripts.oscillo_correlation --signal boitier_1 1
--signal boitier_2 1` depuis la racine du projet (voir src/ui/app.py).
"""

import argparse
import sys

from src.core.config import CorrelationConfig
from src.analysis.correlation_pipeline import CorrelationPipeline


def parse_args():
    d = CorrelationConfig()
    p = argparse.ArgumentParser(description="Pipeline de corrélation inter-signaux.")
    p.add_argument("--project-dir", default=d.project_dir)
    p.add_argument("--signal", nargs=2, metavar=("BOITIER", "VOIE"), action="append",
                   dest="signals",
                   help="Un signal à corréler : boitier puis voie (répéter pour plusieurs). "
                        "Ex: --signal boitier_1 1 --signal boitier_2 1")
    p.add_argument("--output-dir-name", default=d.output_dir_name)
    p.add_argument("--corr-window-s", type=float, default=d.corr_window_s)
    p.add_argument("--hist-bin-min", type=int, default=d.hist_bin_min)
    p.add_argument("--ncc-type-threshold", type=float, default=d.ncc_type_threshold)
    return p.parse_args()


def main():
    args = parse_args()
    default_signals = CorrelationConfig().signals
    signals = (
        [{"boitier": b, "voie": int(v)} for b, v in args.signals]
        if args.signals else default_signals
    )
    config = CorrelationConfig(
        project_dir=args.project_dir,
        signals=signals,
        output_dir_name=args.output_dir_name,
        corr_window_s=args.corr_window_s,
        hist_bin_min=args.hist_bin_min,
        ncc_type_threshold=args.ncc_type_threshold,
    )
    result = CorrelationPipeline(config).run()
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
