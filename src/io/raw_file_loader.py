"""Chargement des fenêtres oscillo depuis des fichiers .txt/.csv bruts."""

import os

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from src.signal_processing.windowing import stack_variable_windows


def _load_one_file(args):
    """Worker : charge un fichier oscillo et retourne ses segments (windows, time_arrays).
    Découpage purement temporel (gap > 0.5s) — pas de troncation à taille fixe."""
    file, channel_index = args
    windows, time_arrs = [], []
    try:
        df = pd.read_csv(file, sep=None, engine='python', header=None)
        df[0] = pd.to_datetime(df[0])
        diffs         = df[0].diff().dt.total_seconds().fillna(0)
        jump_indices  = np.where(diffs > 0.5)[0]
        start_indices = np.concatenate(([0], jump_indices))
        end_indices   = np.concatenate((jump_indices, [len(df)]))
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) <= channel_index:
            return [], []
        val_col = num_cols[channel_index]
        for s, e in zip(start_indices, end_indices):
            window   = df[val_col].iloc[s:e].values
            time_arr = df[0].iloc[s:e].values
            if len(window) < 100:
                continue
            windows.append(window)
            time_arrs.append(time_arr)
    except Exception:
        pass
    return windows, time_arrs


def load_all_oscillo_files(directory="data_", channel_index=0, num_workers=4):
    print(f"--- Analyse du dossier : '{os.path.abspath(directory)}' ---")
    files = sorted([
        os.path.join(root, f)
        for root, _, names in os.walk(directory)
        for f in names
        if f.endswith(".txt") or f.endswith(".csv")
    ])

    all_windows, all_time_arrays = [], []
    args = [(f, channel_index) for f in files]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for windows, time_arrs in executor.map(_load_one_file, args):
            all_windows.extend(windows)
            all_time_arrays.extend(time_arrs)

    print(f"\n--- Chargement terminé : {len(all_windows)} fenêtres récupérées ---")
    return stack_variable_windows(all_windows, all_time_arrays)
