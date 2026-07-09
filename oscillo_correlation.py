#!/usr/bin/env python
# coding: utf-8
"""
Point d'entrée CLI — pipeline de corrélation inter-signaux.

Lit les signal_profile.json produits par oscillo_analysis.py (un dossier par
signal boitier+voie, cf. src/core/paths.py) et détecte les types d'anomalies
partagés entre les signaux sélectionnés.

Usage :
  1. Lancer oscillo_analysis.py pour chaque signal à corréler (un run par
     boitier+voie ; combine autant de jours que voulu via --dates).
  2. python3 oscillo_correlation.py --signal boitier_1 1 --signal boitier_2 1

Zéro argument reproduit exactement le comportement historique (mêmes
signaux/seuils par défaut).
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
