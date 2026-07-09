#!/usr/bin/env python
# coding: utf-8
"""
Point d'entrée CLI — ETL : ingestion des fichiers .txt bruts (oscillo + TRMS)
dans le data lake Parquet partitionné.

La logique vit dans src/ (voir DataLakeWriter). Ce script ne fait que
résoudre la configuration et lancer l'ETL : zéro argument
(`python3 to_data_lake.py`) reproduit exactement l'ancien comportement.
"""

import argparse
import os

from src.core.config import DataLakeConfig
from src.io.datalake_writer import DataLakeWriter


def parse_args():
    d = DataLakeConfig()
    default_workers = min(os.cpu_count() or 1, 5)
    p = argparse.ArgumentParser(
        description="ETL : fichiers .txt bruts -> data lake Parquet partitionné."
    )
    p.add_argument("--source-root-dir", default=d.source_root_dir)
    p.add_argument("--data-lake-dir", default=d.data_lake_dir)
    p.add_argument("--chunk-size-base", type=int, default=d.chunk_size_base)
    p.add_argument("--gpu-sort-threshold", type=int, default=d.gpu_sort_threshold)
    p.add_argument("--n-workers", type=int, default=default_workers)
    return p.parse_args()


def main():
    args = parse_args()
    config = DataLakeConfig(
        source_root_dir=args.source_root_dir,
        data_lake_dir=args.data_lake_dir,
        chunk_size_base=args.chunk_size_base,
        gpu_sort_threshold=args.gpu_sort_threshold,
        n_workers=args.n_workers,
    )
    DataLakeWriter(config).run()


if __name__ == "__main__":
    main()
