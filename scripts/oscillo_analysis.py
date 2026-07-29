#!/usr/bin/env python
# coding: utf-8
"""
Worker CLI interne — un run du pipeline d'analyse oscillo (GPU-accéléré)
pour un seul signal (boitier+voie+dates).

PAS un point d'entrée utilisateur : invoqué en subprocess isolé par
src/analysis/batch_pipeline.py (OscilloBatchRunner), lui-même piloté par
interface.py — le seul point d'entrée du projet. L'isolation en process
séparé (un par job/GPU) permet qu'un pipeline qui plante ne casse pas
l'interface, et libère proprement la VRAM CUDA entre deux runs.

Charge des fenêtres oscillo (data lake Parquet ou fichiers .txt/.csv bruts),
détecte les anomalies par amplitude, les groupe en événements (NCC
temporelle), les clusterise en types (NCC inter-événements), puis exporte un
rapport HTML enrichi + signal_profile.json (consommé par
scripts/oscillo_correlation.py).

La logique du pipeline vit dans src/ (voir OscilloPipeline). Ce script ne
fait que résoudre la configuration (arguments CLI, ou valeurs par défaut) et
lancer le pipeline. Invocation : `python -m scripts.oscillo_analysis ...`
depuis la racine du projet (voir OscilloBatchRunner).
"""

import argparse

from src.core.config import SignalConfig
from src.analysis.oscillo_pipeline import OscilloPipeline


def parse_args():
    d = SignalConfig()
    p = argparse.ArgumentParser(
        description="Pipeline d'analyse oscillo (data lake Parquet ou fichiers bruts)."
    )
    p.add_argument("--use-files", action="store_true",
                   help="Lire depuis des fichiers .txt/.csv bruts (--data-path) "
                        "au lieu du data lake Parquet.")
    p.add_argument("--data-lake-path", default=d.data_lake_path)
    p.add_argument("--boitier", default=d.boitier, help='ex: "boitier_1"')
    p.add_argument("--voie", type=int, default=d.voie)
    p.add_argument("--dates", nargs="+", default=d.dates,
                   help="Une ou plusieurs dates YYYY-MM-DD, combinées dans un seul rapport.")
    p.add_argument("--data-path", default=d.data_path)
    p.add_argument("--output-dir-raw", default=d.output_dir_raw)
    p.add_argument("--channel-index", type=int, default=d.channel_index)
    p.add_argument("--anomaly-n-segments", type=int, default=d.anomaly_n_segments,
                   help="Nombre de segments pour le filtrage d'anomalie par "
                        "amplitude locale (filter_anomaly_windows).")
    p.add_argument("--anomaly-threshold", type=float, default=d.anomaly_threshold,
                   help="Seuil d'amplitude moyenne (max-min par segment) au-delà "
                        "duquel une fenêtre est une anomalie.")
    p.add_argument("--peak-threshold", type=float, default=d.peak_threshold,
                   help="Seuil sur les pics de la fenêtre entière : anomalie "
                        "seulement si max >= seuil ET |min| >= seuil.")
    p.add_argument("--lowpass-cutoff-hz", type=float, default=d.lowpass_cutoff_hz,
                   help="Fréquence de coupure (Hz) du filtre passe-bas appliqué "
                        "avant le calcul de similarité NCC.")
    p.add_argument("--ncc-threshold", type=float, default=d.ncc_threshold)
    p.add_argument("--ncc-max-lag", type=int, default=d.ncc_max_lag)
    p.add_argument("--ncc-type-threshold", type=float, default=d.ncc_type_threshold)
    p.add_argument("--n-freq-bins", type=int, default=d.n_freq_bins)
    p.add_argument("--gpu-batch-size", type=int, default=d.gpu_batch_size)
    p.add_argument("--freq-chunk", type=int, default=d.freq_chunk)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--type-hist-bin-min", type=int, default=d.type_hist_bin_min,
                   help="Largeur des bins (minutes) de l'histogramme d'apparition "
                        "par type dans rapport_types_detaille.html.")
    p.add_argument("--type-db-dir", default=d.type_db_dir,
                   help="Dossier de la base de types persistante (globale, "
                        "jamais effacée entre les runs).")
    p.add_argument("--global-type-ncc-threshold", type=float,
                   default=d.global_type_ncc_threshold,
                   help="Seuil NCC (signal brut, sans filtre passe-bas) pour "
                        "rattacher un type local à un type déjà connu de la base.")
    return p.parse_args()


def main():
    args = parse_args()
    config = SignalConfig(
        use_datalake=not args.use_files,
        data_lake_path=args.data_lake_path,
        boitier=args.boitier,
        voie=args.voie,
        dates=args.dates,
        data_path=args.data_path,
        output_dir_raw=args.output_dir_raw,
        channel_index=args.channel_index,
        anomaly_n_segments=args.anomaly_n_segments,
        anomaly_threshold=args.anomaly_threshold,
        peak_threshold=args.peak_threshold,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        ncc_threshold=args.ncc_threshold,
        ncc_max_lag=args.ncc_max_lag,
        ncc_type_threshold=args.ncc_type_threshold,
        n_freq_bins=args.n_freq_bins,
        gpu_batch_size=args.gpu_batch_size,
        freq_chunk=args.freq_chunk,
        num_workers=args.num_workers,
        type_hist_bin_min=args.type_hist_bin_min,
        type_db_dir=args.type_db_dir,
        global_type_ncc_threshold=args.global_type_ncc_threshold,
    )
    OscilloPipeline(config).run()


if __name__ == "__main__":
    main()
