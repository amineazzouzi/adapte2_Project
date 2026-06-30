"""Bibliothèque persistante de types d'anomalies inter-runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..signal.ncc import compute_ncc_single


class TypeLibrary:
    """Bibliothèque persistante comparant les types d'anomalies entre runs."""

    def __init__(self, library_dir: Path) -> None:
        self.library_dir = Path(library_dir)
        self._metadata: dict[int, dict] = {}
        self._windows: dict[int, np.ndarray] = {}

    @classmethod
    def load(cls, library_dir: Path) -> "TypeLibrary":
        """Charge metadata.json + tous les type_*.npy depuis library_dir."""
        lib = cls(library_dir)
        meta_path = lib.library_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                lib._metadata = {int(k): v for k, v in json.load(f).items()}
            for type_id in lib._metadata:
                npy = lib.library_dir / f"type_{type_id}_ref.npy"
                if npy.exists():
                    lib._windows[type_id] = np.load(str(npy))
        return lib

    def match_or_register(
        self,
        ref_window: np.ndarray,
        dom_freq: float,
        signal_id: str,
        date: str,
        ncc_threshold: float = 0.6,
    ) -> tuple[int, bool]:
        """Compare ref_window contre tous les types connus.

        Retourne (type_id, is_new). Si NCC >= threshold → type existant.
        Sinon → crée et retourne un nouveau type.
        """
        w = np.asarray(ref_window, dtype=np.float64)
        best_id = -1
        best_ncc = -1.0

        for type_id, known_w in self._windows.items():
            n = min(len(w), len(known_w))
            ncc = compute_ncc_single(w[:n], known_w[:n])
            if ncc > best_ncc:
                best_ncc = ncc
                best_id = type_id

        if best_ncc >= ncc_threshold:
            return best_id, False

        # Nouveau type
        new_id = max(self._metadata.keys(), default=-1) + 1
        self._windows[new_id] = w.astype(np.float32)
        self._metadata[new_id] = {
            "first_seen": date,
            "last_seen": date,
            "total_occurrences": 0,
            "mean_freq": dom_freq,
            "signal_ids": [],
        }
        return new_id, True

    def update_occurrence(
        self,
        type_id: int,
        signal_id: str,
        date: str,
        dom_freq: float,
        event_count: int,
    ) -> None:
        """Ajoute une occurrence au type."""
        if type_id not in self._metadata:
            return
        meta = self._metadata[type_id]
        meta["total_occurrences"] += event_count
        meta["last_seen"] = date
        if signal_id not in meta["signal_ids"]:
            meta["signal_ids"].append(signal_id)
        n = meta["total_occurrences"]
        meta["mean_freq"] = (meta["mean_freq"] * (n - event_count) + dom_freq * event_count) / n

    def save(self) -> None:
        """Écrit metadata.json + .npy files dans library_dir."""
        self.library_dir.mkdir(parents=True, exist_ok=True)
        with open(self.library_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        for type_id, w in self._windows.items():
            np.save(str(self.library_dir / f"type_{type_id}_ref.npy"), w)

    def get_type_summary(self) -> pd.DataFrame:
        """Retourne un DataFrame résumant tous les types connus."""
        rows = []
        for type_id, meta in self._metadata.items():
            rows.append(
                {
                    "type_id": type_id,
                    "first_seen": meta.get("first_seen"),
                    "last_seen": meta.get("last_seen"),
                    "total_occurrences": meta.get("total_occurrences", 0),
                    "mean_freq": meta.get("mean_freq", 0.0),
                    "n_signals": len(meta.get("signal_ids", [])),
                }
            )
        return pd.DataFrame(rows)
