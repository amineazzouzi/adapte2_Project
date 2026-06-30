"""Orchestrateur principal du pipeline d'analyse oscilloscope."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import PipelineConfig
from ..io.data_lake import DataLakeReader
from ..io.file_loader import load_oscillo_files
from ..signal.filtering import filter_by_amplitude
from ..signal.features import compute_feature_matrix
from .clustering import group_events_by_ncc, cluster_events_by_type, precompute_metrics
from .type_library import TypeLibrary


class OscilloPipeline:
    """Orchestrateur du pipeline d'analyse oscilloscope Adapte2."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        type_library: TypeLibrary | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.type_library = type_library

    def run_from_datalake(
        self,
        boitier: str,
        voie: int,
        date: str,
        output_dir: Path,
    ) -> dict:
        """Charge depuis le data lake et exécute le pipeline complet."""
        reader = DataLakeReader(self.config.paths.data_lake)
        windows, time_arrays = reader.load_day(
            boitier, voie, date, self.config.signal.target_pts
        )
        signal_id = f"{boitier}/voie_{voie}/{date}"
        return self._run(windows, time_arrays, signal_id, output_dir)

    def run_from_files(
        self,
        file_paths: list[Path],
        output_dir: Path,
        channel_index: int = 0,
        signal_id: str = "signal",
    ) -> dict:
        """Charge depuis des fichiers .txt/.csv et exécute le pipeline complet."""
        windows, time_arrays = load_oscillo_files(
            file_paths,
            channel_index=channel_index,
            target_pts=self.config.signal.target_pts,
            num_workers=self.config.num_workers,
        )
        return self._run(windows, time_arrays, signal_id, output_dir)

    def _run(
        self,
        windows: np.ndarray,
        time_arrays: np.ndarray,
        signal_id: str,
        output_dir: Path,
    ) -> dict:
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        cfg = self.config.signal

        if len(windows) == 0:
            print("Aucune fenêtre chargée — pipeline interrompu.")
            return {}

        # 1. Filtrage anomalies
        t0 = time.time()
        is_anomaly = filter_by_amplitude(windows, cfg.n_segments, cfg.amplitude_threshold)
        n_anom = int(np.sum(is_anomaly))
        print(
            f"Anomalies : {n_anom} / {len(is_anomaly)} "
            f"(seuil amplitude > {cfg.amplitude_threshold})"
        )
        print(f"  Filtrage : {time.time()-t0:.2f}s")

        timestamps = [ta[0] for ta in time_arrays]

        # 2. Groupement NCC
        t0 = time.time()
        events = group_events_by_ncc(
            windows, is_anomaly, timestamps,
            threshold=cfg.ncc_threshold,
            max_lag=cfg.ncc_max_lag,
        )
        print(f"  {len(events)} événements groupés  [{time.time()-t0:.2f}s]")

        if not events:
            print("Aucun événement détecté.")
            return {}

        # 3. Pré-calcul métriques
        t0 = time.time()
        dom_freq_map, ncc_map, is_ref_map = precompute_metrics(
            events, windows, time_arrays, cfg.ncc_max_lag
        )
        print(f"  Pré-calcul : {time.time()-t0:.2f}s")

        # 4. Features enrichies (sur les fenêtres d'anomalie uniquement)
        anom_idx = np.array([i for ev in events for i in ev["indices"]])
        if len(anom_idx) > 0:
            feature_df = compute_feature_matrix(windows[anom_idx], time_arrays[anom_idx])
        else:
            feature_df = pd.DataFrame()

        # 5. Clustering inter-événements
        t0 = time.time()
        cluster_labels, ncc_matrix = cluster_events_by_type(
            events, windows,
            ncc_type_threshold=cfg.ncc_type_threshold,
            ncc_max_lag=cfg.ncc_max_lag,
            gpu_batch_size=self.config.gpu.batch_size,
        )
        n_types = max(cluster_labels, default=-1) + 1
        print(f"  {len(events)} événements → {n_types} types  [{time.time()-t0:.2f}s]")

        # 6. Mapping TypeLibrary si fournie
        if self.type_library is not None:
            for ev_idx, event in enumerate(events):
                ref_idx = event["indices"][0]
                ref_w = windows[ref_idx]
                freq = float(dom_freq_map.get(ref_idx, 0.0))
                date_str = str(signal_id).split("/")[-1] if "/" in signal_id else ""
                global_id, is_new = self.type_library.match_or_register(
                    ref_w, freq, signal_id, date_str, ncc_threshold=0.6
                )
                self.type_library.update_occurrence(
                    global_id, signal_id, date_str, freq, 1
                )
            self.type_library.save()

        # 7. Construction du profil
        profile = self._build_signal_profile(
            events, windows, time_arrays,
            dom_freq_map, ncc_map, cluster_labels, signal_id,
        )
        profile["feature_summary"] = (
            feature_df.describe().to_dict() if not feature_df.empty else {}
        )

        # 8. Sauvegarde
        self.save_profile(profile, output_dir)
        self._save_type_windows(profile, output_dir)

        return profile

    def _build_signal_profile(
        self,
        events: list[dict],
        windows: np.ndarray,
        time_arrays: np.ndarray,
        dom_freq_map: dict,
        ncc_map: dict,
        cluster_labels: list[int],
        signal_id: str,
    ) -> dict:
        events_out = []
        for ev_idx, event in enumerate(events):
            indices = event["indices"]
            ref_idx = indices[0]
            win_ts = [pd.Timestamp(time_arrays[i][0]) for i in indices]
            debut_ts = pd.Timestamp(event["debut"])
            fin_ts = pd.Timestamp(event["fin"])
            duration = max((fin_ts - debut_ts).total_seconds(), 0.0)

            events_out.append(
                {
                    "event_id": ev_idx,
                    "cluster_id": cluster_labels[ev_idx],
                    "ref_win_idx": ref_idx,
                    "ref_timestamp": win_ts[0],
                    "ref_window": windows[ref_idx],
                    "ref_dom_freq": float(dom_freq_map.get(ref_idx, 0.0)),
                    "window_timestamps": win_ts,
                    "ncc_vs_ref": [float(ncc_map.get(i, 1.0)) for i in indices],
                    "dom_freqs": [float(dom_freq_map.get(i, 0.0)) for i in indices],
                    "window_count": len(indices),
                    "duration_s": duration,
                }
            )

        t_start = min(e["ref_timestamp"] for e in events_out) if events_out else None
        t_end = max(e["window_timestamps"][-1] for e in events_out) if events_out else None
        n_clust = max(cluster_labels, default=-1) + 1

        return {
            "signal_id": signal_id,
            "events": events_out,
            "total_windows": sum(e["window_count"] for e in events_out),
            "t_start": t_start,
            "t_end": t_end,
            "n_clusters": n_clust,
        }

    def save_profile(self, signal_profile: dict, output_dir: Path) -> str:
        """Sérialise signal_profile.json (sans les arrays numpy)."""

        def _ts(t: object) -> str | None:
            return pd.Timestamp(t).isoformat() if t is not None else None

        data = {
            "signal_id": signal_profile["signal_id"],
            "total_windows": signal_profile["total_windows"],
            "n_clusters": signal_profile["n_clusters"],
            "t_start": _ts(signal_profile["t_start"]),
            "t_end": _ts(signal_profile["t_end"]),
            "events": [
                {
                    "event_id": ev["event_id"],
                    "cluster_id": ev["cluster_id"],
                    "ref_timestamp": _ts(ev["ref_timestamp"]),
                    "ref_dom_freq": ev["ref_dom_freq"],
                    "window_count": ev["window_count"],
                    "duration_s": ev["duration_s"],
                    "window_timestamps": [_ts(t) for t in ev["window_timestamps"]],
                    "dom_freqs": ev["dom_freqs"],
                    "ncc_vs_ref": ev["ncc_vs_ref"],
                }
                for ev in signal_profile["events"]
            ],
        }

        path = output_dir / "signal_profile.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"--- Profil JSON : '{path}' ---")
        return str(path)

    def _save_type_windows(self, signal_profile: dict, output_dir: Path) -> None:
        """Sauvegarde la fenêtre de référence du premier événement de chaque type."""
        refs_dir = output_dir / "refs"
        refs_dir.mkdir(exist_ok=True)
        seen: set[int] = set()
        for ev in signal_profile["events"]:
            cid = ev["cluster_id"]
            if cid in seen:
                continue
            seen.add(cid)
            arr = np.asarray(ev["ref_window"], dtype=np.float32)
            np.save(str(refs_dir / f"type{cid}_window.npy"), arr)
        print(f"  -> {len(seen)} fenêtre(s) de type sauvegardée(s) dans 'refs/'")
