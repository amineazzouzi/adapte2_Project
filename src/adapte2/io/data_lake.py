"""Lecture du data lake Parquet partitionné par boîtier/voie/date."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..core.config import PipelineConfig


class DataLakeReader:
    """Accès en lecture au data lake Parquet."""

    def __init__(self, data_lake_path: str) -> None:
        self.root = Path(data_lake_path)

    def load_day(
        self,
        boitier: str,
        voie: int,
        date: str,
        target_pts: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Charge un jour Parquet et segmente en fenêtres de target_pts points.

        Retourne (windows, time_arrays) de shapes (N, target_pts).
        """
        year, month, day = date.split("-")
        parquet_path = (
            self.root / boitier / "oscillo" / f"voie_{voie}"
            / f"year={year}" / f"month={int(month):02d}" / f"day={int(day):02d}"
            / "data.parquet"
        )

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet introuvable : {parquet_path}")

        print(f"--- Chargement data lake : {parquet_path} ---")
        df = pd.read_parquet(parquet_path)
        df = df.sort_values("timestamp").reset_index(drop=True)

        diffs = df["timestamp"].diff().dt.total_seconds().fillna(0)
        jump_indices = np.where(diffs > 0.5)[0]
        start_indices = np.concatenate(([0], jump_indices))
        end_indices = np.concatenate((jump_indices, [len(df)]))

        all_windows: list[np.ndarray] = []
        all_time_arrays: list[np.ndarray] = []

        for s, e in zip(start_indices, end_indices):
            seg_sig = df["signal"].iloc[s:e].values.astype(np.float64)
            seg_ts = df["timestamp"].iloc[s:e].values

            if len(seg_sig) < 100:
                continue

            seg_sig, seg_ts = _pad_or_truncate(seg_sig, seg_ts, target_pts)
            all_windows.append(seg_sig)
            all_time_arrays.append(seg_ts)

        n = len(all_windows)
        print(f"--- {n} fenêtres chargées ({boitier} / voie_{voie} / {date}) ---")
        return np.array(all_windows), np.array(all_time_arrays, dtype=object)

    def list_available_dates(self, boitier: str, voie: int) -> list[str]:
        """Retourne les dates disponibles au format YYYY-MM-DD."""
        base = self.root / boitier / "oscillo" / f"voie_{voie}"
        if not base.exists():
            return []
        dates: list[str] = []
        for year_dir in sorted(base.glob("year=*")):
            year = year_dir.name.split("=")[1]
            for month_dir in sorted(year_dir.glob("month=*")):
                month = month_dir.name.split("=")[1]
                for day_dir in sorted(month_dir.glob("day=*")):
                    day = day_dir.name.split("=")[1]
                    if (day_dir / "data.parquet").exists():
                        dates.append(f"{year}-{int(month):02d}-{int(day):02d}")
        return dates

    def list_boitiers(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def list_voies(self, boitier: str) -> list[int]:
        base = self.root / boitier / "oscillo"
        if not base.exists():
            return []
        voies = []
        for p in sorted(base.glob("voie_*")):
            try:
                voies.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return voies


def _pad_or_truncate(
    signal: np.ndarray,
    timestamps: np.ndarray,
    target_pts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Padde ou tronque signal et timestamps à target_pts points."""
    n = len(signal)
    if n < target_pts:
        pad = target_pts - n
        signal = np.pad(signal, (0, pad), "constant")
        if len(timestamps) > 1:
            dt = timestamps[1] - timestamps[0]
            extra = np.array([timestamps[-1] + (k + 1) * dt for k in range(pad)])
        else:
            extra = np.array([timestamps[-1]] * pad)
        timestamps = np.concatenate([timestamps, extra])
    else:
        signal = signal[:target_pts]
        timestamps = timestamps[:target_pts]
    return signal, timestamps
