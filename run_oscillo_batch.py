#!/usr/bin/env python
# coding: utf-8
"""
Point d'entrée CLI — dispatch de plusieurs runs oscillo_analysis.py en
parallèle sur un serveur multi-GPU (un job = un boitier/voie/dates, un GPU
par job en round-robin via CUDA_VISIBLE_DEVICES).

La liste des jobs est explicite : soit éditée à la main dans
BatchConfig.jobs (src/core/config.py), soit passée en CLI via --job répété.

Usage :
  python3 run_oscillo_batch.py
  python3 run_oscillo_batch.py \\
      --job boitier_1 1 2026-03-17 \\
      --job boitier_1 2 2026-03-17 \\
      --job boitier_2 1 2026-03-17

trms_analysis.py n'est pas concerné par ce dispatcher.
"""

import argparse
import os

from src.core.config import BatchConfig
from src.analysis.batch_pipeline import OscilloBatchRunner


def parse_args():
    d = BatchConfig()
    p = argparse.ArgumentParser(
        description="Dispatch de plusieurs runs oscillo_analysis.py en parallèle (multi-GPU)."
    )
    p.add_argument("--data-lake-path", default=d.data_lake_path)
    p.add_argument(
        "--job", nargs="+", action="append", dest="jobs",
        metavar="BOITIER VOIE DATE [DATE ...]",
        help="Un job : boitier, voie, une ou plusieurs dates. Répéter pour "
             "plusieurs jobs. Ex: --job boitier_1 1 2026-03-17"
    )
    p.add_argument("--max-parallel", type=int, default=d.max_parallel,
                   help="Nombre de jobs simultanés (0 = auto, un par GPU détecté).")
    p.add_argument("--log-dir", default=d.log_dir)
    return p.parse_args()


def main():
    args = parse_args()

    if args.jobs:
        jobs = [
            {"boitier": j[0], "voie": int(j[1]), "dates": j[2:]}
            for j in args.jobs
        ]
    else:
        jobs = BatchConfig().jobs

    config = BatchConfig(
        data_lake_path=args.data_lake_path,
        jobs=jobs,
        max_parallel=args.max_parallel,
        log_dir=args.log_dir,
    )

    project_dir = os.path.dirname(os.path.abspath(__file__))
    OscilloBatchRunner(config, project_dir).run()


if __name__ == "__main__":
    main()
