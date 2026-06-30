"""Chargement de fichiers .txt/.csv oscilloscopes bruts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data_lake import _pad_or_truncate


def scan_directory(directory: Path) -> list[Path]:
    """Retourne tous les .txt et .csv récursivement."""
    return sorted(
        p for p in directory.rglob("*")
        if p.suffix.lower() in (".txt", ".csv") and p.is_file()
    )


def load_oscillo_files(
    paths: list[Path],
    channel_index: int = 0,
    target_pts: int = 5000,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Charge une liste de fichiers oscillo et retourne (windows, time_arrays)."""
    args = [(p, channel_index, target_pts) for p in paths]
    all_windows: list[np.ndarray] = []
    all_time_arrays: list[np.ndarray] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for windows, time_arrs in executor.map(_load_one_file, args):
            all_windows.extend(windows)
            all_time_arrays.extend(time_arrs)

    print(f"--- Chargement terminé : {len(all_windows)} fenêtres récupérées ---")
    if not all_windows:
        return np.empty((0, target_pts)), np.empty((0, target_pts), dtype=object)
    return np.array(all_windows), np.array(all_time_arrays, dtype=object)


def _load_one_file(
    args: tuple[Path, int, int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Worker : charge un fichier et retourne ses segments (windows, time_arrays)."""
    file, channel_index, target_pts = args
    windows: list[np.ndarray] = []
    time_arrs: list[np.ndarray] = []
    try:
        df = pd.read_csv(file, sep=None, engine="python", header=None)
        df[0] = pd.to_datetime(df[0])
        diffs = df[0].diff().dt.total_seconds().fillna(0)
        jump_indices = np.where(diffs > 0.5)[0]
        start_indices = np.concatenate(([0], jump_indices))
        end_indices = np.concatenate((jump_indices, [len(df)]))

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) <= channel_index:
            return [], []
        val_col = num_cols[channel_index]

        for s, e in zip(start_indices, end_indices):
            window = df[val_col].iloc[s:e].values
            time_arr = df[0].iloc[s:e].values
            if len(window) < 100:
                continue
            window, time_arr = _pad_or_truncate(window, time_arr, target_pts)
            windows.append(window)
            time_arrs.append(time_arr)
    except Exception:
        pass
    return windows, time_arrs
