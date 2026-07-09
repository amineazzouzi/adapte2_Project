#!/usr/bin/env python
# coding: utf-8
"""
Point d'entrée CLI unique — enchaîne oscillo_analysis puis oscillo_correlation
sur une ou plusieurs sources (signaux), sans passer par l'interface graphique
(interface.py) ni par des appels subprocess : les pipelines sont appelés
directement en Python, adapté à une exécution headless (ex: job SLURM).

Usage :
  python3 main.py --signal boitier_1 1 2026-03-17 2026-03-18 \\
                   --signal boitier_2 1 2026-03-17

Chaque --signal prend : BOITIER VOIE DATE [DATE ...] (répéter l'option pour
ajouter d'autres sources). Toutes les dates d'un même signal sont combinées
dans un seul rapport (comme oscillo_analysis.py --dates). La corrélation
inter-signaux n'est lancée que si au moins 2 sources sont fournies.

Zéro argument reproduit le comportement historique par défaut (les signaux
de CorrelationConfig().signals, chacun sur SignalConfig().dates).
"""

import argparse
import sys

from src.core.config import SignalConfig, CorrelationConfig
from src.analysis.oscillo_pipeline import OscilloPipeline
from src.analysis.correlation_pipeline import CorrelationPipeline


def parse_args():
    d = SignalConfig()
    dc = CorrelationConfig()
    p = argparse.ArgumentParser(
        description="Pipeline complet (analyse + corrélation) sur une ou plusieurs sources."
    )
    p.add_argument(
        "--signal", nargs="+", metavar="BOITIER VOIE DATE [DATE ...]",
        action="append", dest="signals",
        help="Une source à analyser : boitier, voie, puis une ou plusieurs dates "
             "YYYY-MM-DD. Répéter l'option pour plusieurs sources. "
             "Ex: --signal boitier_1 1 2026-03-17 2026-03-18",
    )
    p.add_argument("--data-lake-path", default=d.data_lake_path)
    p.add_argument("--ncc-threshold", type=float, default=d.ncc_threshold)
    p.add_argument("--ncc-max-lag", type=int, default=d.ncc_max_lag)
    p.add_argument("--ncc-type-threshold", type=float, default=d.ncc_type_threshold)
    p.add_argument("--n-freq-bins", type=int, default=d.n_freq_bins)
    p.add_argument("--gpu-batch-size", type=int, default=d.gpu_batch_size)
    p.add_argument("--freq-chunk", type=int, default=d.freq_chunk)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--corr-window-s", type=float, default=dc.corr_window_s)
    p.add_argument("--hist-bin-min", type=int, default=dc.hist_bin_min)
    p.add_argument("--corr-ncc-type-threshold", type=float, default=dc.ncc_type_threshold)
    return p.parse_args()


def resolve_jobs(args):
    if args.signals:
        return [
            {"boitier": s[0], "voie": int(s[1]), "dates": s[2:] or SignalConfig().dates}
            for s in args.signals
        ]
    return [
        {**sig, "dates": SignalConfig().dates}
        for sig in CorrelationConfig().signals
    ]


def main():
    args = parse_args()
    jobs = resolve_jobs(args)

    print(f"\n{'='*68}\nPIPELINE ADAPTE2 — {len(jobs)} source(s) à analyser\n{'='*68}\n")

    for i, job in enumerate(jobs, 1):
        date_label = job["dates"][0] if len(job["dates"]) == 1 else f"{job['dates'][0]} → {job['dates'][-1]}"
        print(f"\n[{i}/{len(jobs)}] Analyse : {job['boitier']} / voie_{job['voie']} / {date_label}\n")
        config = SignalConfig(
            data_lake_path=args.data_lake_path,
            boitier=job["boitier"],
            voie=job["voie"],
            dates=job["dates"],
            ncc_threshold=args.ncc_threshold,
            ncc_max_lag=args.ncc_max_lag,
            ncc_type_threshold=args.ncc_type_threshold,
            n_freq_bins=args.n_freq_bins,
            gpu_batch_size=args.gpu_batch_size,
            freq_chunk=args.freq_chunk,
            num_workers=args.num_workers,
        )
        OscilloPipeline(config).run()

    if len(jobs) >= 2:
        print(f"\n{'='*68}\nCORRÉLATION — {len(jobs)} source(s)\n{'='*68}\n")
        corr_config = CorrelationConfig(
            signals=[{"boitier": j["boitier"], "voie": j["voie"]} for j in jobs],
            corr_window_s=args.corr_window_s,
            hist_bin_min=args.hist_bin_min,
            ncc_type_threshold=args.corr_ncc_type_threshold,
        )
        result = CorrelationPipeline(corr_config).run()
        if result is None:
            sys.exit(1)
    else:
        print("\nCorrélation ignorée (moins de 2 sources).\n")


if __name__ == "__main__":
    main()
